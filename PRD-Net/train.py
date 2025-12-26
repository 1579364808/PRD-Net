import os
import torch
import yaml
import argparse
from core.dataset import MMDataLoader
from core.losses import PRD_Loss
from core.scheduler import get_scheduler
from core.utils import setup_seed, get_best_results, interval_time
from models.prd_net import build_model
from core.metric import MetricsTop
from core.config_utils import parse_args_with_overrides, apply_overrides
from tqdm import tqdm
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

# 创建参数解析器（支持动态覆盖）
parser = argparse.ArgumentParser(
    description='训练脚本，支持命令行参数覆盖配置文件',
    epilog='''
示例:
  python train.py --config_file configs/train_sims.yaml
  python train.py --config_file configs/train_sims.yaml --model.prm.temperature=0.1
  python train.py --config_file configs/train_sims.yaml --base.lr=0.0001 --base.n_epochs=50
'''
)
parser.add_argument('--config_file', type=str, default='configs/train_mosi_exp3.yaml')
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--ckpt_dir', type=str, default=None, help='自定义checkpoint保存目录')
opt, overrides = parse_args_with_overrides(parser)
print(opt)
if overrides:
    print(f"命令行覆盖参数: {overrides}")

def main():
    best_valid_results, best_test_results = {}, {}

    config_file = opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # 应用命令行覆盖参数
    if overrides:
        args = apply_overrides(args, overrides)
        print(f"应用覆盖后的配置:")
    print(args)

    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)
    print("seed is fixed to {}".format(seed))

    # 使用自定义ckpt目录或默认目录
    if opt.ckpt_dir:
        ckpt_root = opt.ckpt_dir
    else:
        ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])
    os.makedirs(ckpt_root, exist_ok=True)
    print("ckpt root :", ckpt_root)

    model = build_model(args).to(device)

    dataLoader = MMDataLoader(args)

    # ============ 分层学习率设置 ============
    # BERT作为预训练模型，使用较小的学习率
    # 其他新初始化的层使用较大的学习率
    bert_lr = args['base'].get('bert_lr', args['base']['lr'] * 0.1)  # 默认为主学习率的0.1
    main_lr = args['base']['lr']
    weight_decay = args['base']['weight_decay']

    # 获取BERT参数的ID
    bert_params_id = list(map(id, model.bertmodel.parameters()))

    # 分离参数组
    bert_params = filter(lambda p: id(p) in bert_params_id, model.parameters())
    other_params = filter(lambda p: id(p) not in bert_params_id, model.parameters())

    # 创建参数组
    param_groups = [
        {'params': list(bert_params), 'lr': bert_lr, 'name': 'bert'},
        {'params': list(other_params), 'lr': main_lr, 'name': 'other'},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    print(f"[分层学习率] BERT lr: {bert_lr}, Other lr: {main_lr}")

    scheduler_warmup = get_scheduler(optimizer, args)

    loss_fn = PRD_Loss(args)

    metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(args['dataset']['datasetName'])


    for epoch in range(1, args['base']['n_epochs']+1):
        print(f'Training Epoch: {epoch}')
        start_time = time.time()
        train_loader = tqdm(dataLoader['train'], total=len(dataLoader['train']))
        train(model, train_loader, optimizer, loss_fn, epoch, metrics, args)

        if args['base']['do_validation']:
            valid_results = evaluate(model, dataLoader['valid'], loss_fn, epoch, metrics)
            best_valid_results = get_best_results(valid_results, best_valid_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=False)
            print(f'Current Best Valid Results: {best_valid_results}')

        test_results = evaluate(model, dataLoader['test'], loss_fn, epoch, metrics)
        best_test_results = get_best_results(test_results, best_test_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=True)
        print(f'Current Best Test Results: {best_test_results}\n')

        end_time = time.time()
        epoch_mins, epoch_secs = interval_time(start_time, end_time)
        print(f'Time: {epoch_mins}m {epoch_secs}s')

        scheduler_warmup.step()


def train(model, train_loader, optimizer, loss_fn, epoch, metrics, args):
    y_pred, y_true = [], []
    loss_dict = {}

    # 梯度累积参数
    accumulation_steps = args['base'].get('gradient_accumulation_steps', 1)

    # 更新loss_fn的当前epoch（用于β warmup）
    loss_fn.set_epoch(epoch)

    model.train()
    optimizer.zero_grad()  # 在epoch开始时清零梯度

    for cur_iter, data in enumerate(train_loader):
        complete_input = (data['vision'].to(device), data['audio'].to(device), data['text'].to(device))
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)

        # 三模态完整性标签（用于监督完整性估计器）
        completeness_labels_l = 1. - data['labels']['missing_rate_l'].to(device)
        completeness_labels_a = 1. - data['labels']['missing_rate_a'].to(device)
        completeness_labels_v = 1. - data['labels']['missing_rate_v'].to(device)

        label = {
            'sentiment_labels': sentiment_labels,
            'completeness_labels_l': completeness_labels_l,
            'completeness_labels_a': completeness_labels_a,
            'completeness_labels_v': completeness_labels_v,
        }

        # 传递情感标签给模型，用于改进CNC中的对比学习

        out = model(complete_input, incomplete_input)

        loss = loss_fn(out, label)

        # 梯度累积：损失除以累积步数
        scaled_loss = loss['loss'] / accumulation_steps
        scaled_loss.backward()

        # ============ PRM码本更新（必须在backward之后） ============
        # 这样做是为了避免forward中的inplace操作导致梯度计算失败

        # ⚠️ 极度重要：必须传入完整数据特征更新码本！
        # out['complete_l/a/v'] 来自 complete_input（完整数据）
        # 绝不能使用 incomplete_input（被mask污染的数据）
        if args['base'].get('use_prm', False) and out.get('complete_l') is not None:
            with torch.no_grad():
                model.prm.update_codebook(
                    out['complete_l'].detach(),  # 完整数据特征，用于更新码本
                    out['complete_a'].detach(),
                    out['complete_v'].detach()
                )

        # 每accumulation_steps步或最后一个batch时更新参数
        if (cur_iter + 1) % accumulation_steps == 0 or (cur_iter + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                loss_dict[key] = value.item() if hasattr(value, 'item') else value
        else:
            for key, value in loss.items():
                loss_dict[key] += value.item() if hasattr(value, 'item') else value

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    loss_dict = {key: value / (cur_iter+1) for key, value in loss_dict.items()}

    print(f'Train Loss Epoch {epoch}: {loss_dict}')
    print(f'Train Results Epoch {epoch}: {results}')
    if accumulation_steps > 1:
        print(f'  [Gradient Accumulation: {accumulation_steps} steps, Effective Batch Size: {args["base"]["batch_size"] * accumulation_steps}]')




def evaluate(model, eval_loader, loss_fn, epoch, metrics):
    loss_dict = {}

    y_pred, y_true = [], []

    model.eval()

    eval_loader = tqdm(eval_loader, total=len(eval_loader), desc="Evaluating")

    for cur_iter, data in enumerate(eval_loader):
        complete_input = (None, None, None)
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)

        # 三模态完整性标签（评估时也需要，用于计算 l_cc）
        completeness_labels_l = 1. - data['labels']['missing_rate_l'].to(device)
        completeness_labels_a = 1. - data['labels']['missing_rate_a'].to(device)
        completeness_labels_v = 1. - data['labels']['missing_rate_v'].to(device)

        label = {
            'sentiment_labels': sentiment_labels,
            'completeness_labels_l': completeness_labels_l,
            'completeness_labels_a': completeness_labels_a,
            'completeness_labels_v': completeness_labels_v,
        }

        with torch.no_grad():
            out = model(complete_input, incomplete_input)

        loss = loss_fn(out, label)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                try:
                    loss_dict[key] = value.item()
                except:
                    loss_dict[key] = value
        else:
            for key, value in loss.items():
                try:
                    loss_dict[key] += value.item()
                except:
                    loss_dict[key] += value

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    return results


if __name__ == '__main__':
    main()


