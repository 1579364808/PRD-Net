import os
import torch
import argparse
import copy
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, dict_to_namespace, get_best_results, interval_time
# from tensorboardX import SummaryWriter
from models.almt import build_model
from core.metric import MetricsTop
import yaml
from core.config_utils import parse_args_with_overrides, apply_overrides
from tqdm import tqdm
import time


# parser = argparse.ArgumentParser() 
# parser.add_argument('--config_file', type=str, default='configs/sims.yaml') 
# parser.add_argument('--seed', type=int, default=-1) 
# parser.add_argument('--gpu_id', type=int, default=-1) 
# opt = parser.parse_args()
# print(opt)

# 创建参数解析器（支持动态覆盖）
parser = argparse.ArgumentParser(
    description='训练脚本，支持命令行参数覆盖配置文件',
    epilog='''
示例:
  python train.py --config_file configs/sims.yaml
  python train.py --config_file configs/sims.yaml --base.lr=0.0001
'''
)
parser.add_argument('--config_file', type=str, default='configs/sims.yaml') 
parser.add_argument('--seed', type=int, default=-1) 
parser.add_argument('--gpu_id', type=int, default=-1) 
opt, overrides = parse_args_with_overrides(parser)
print(opt)
if overrides:
    print(f"命令行覆盖参数: {overrides}")

with open(opt.config_file) as f:
    args_dict = yaml.load(f, Loader=yaml.FullLoader)

# Apply overrides to the dictionary
if overrides:
    args_dict = apply_overrides(args_dict, overrides)
    print(f"应用覆盖后的配置:")

args = dict_to_namespace(copy.deepcopy(args_dict))
print(args)

seed = args.base.seed if opt.seed == -1 else opt.seed
# gpu_id = args.base.gpu_id if opt.gpu_id == -1 else opt.gpu_id

print('-----------------args-----------------')
print(args)
print('-------------------------------------')

# gpu_id = str(gpu_id)
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Device: {device}")


def main():
    setup_seed(seed)
    log_path = os.path.join(".", "log", args.base.project_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    save_path = os.path.join(args.base.ckpt_root, args.base.project_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Ensure ckpt directory exists for robust evaluation
    ckpt_dir = os.path.join('ckpt', args.dataset.datasetName)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model = build_model(args).to(device)

    dataLoader = MMDataLoader(args_dict)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.base.lr,
                                 weight_decay=args.base.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, args)

    loss_fn = torch.nn.MSELoss()

    metrics_fn = MetricsTop().getMetics(args.dataset.datasetName)

    best_valid_results, best_test_results = {}, {}

    # writer = SummaryWriter(logdir=log_path)

    for epoch in range(1, args.base.n_epochs+1):
        print(f'Training Epoch: {epoch}')
        start_time = time.time()
        train_loader = tqdm(dataLoader['train'], total=len(dataLoader['train']))
        training_ret = train(model, train_loader, optimizer, loss_fn, metrics_fn)
        
        validation_ret = {'results': {}}
        if args.base.do_validation:
            validation_ret = evaluate(model, dataLoader['valid'], loss_fn, metrics_fn)
            best_valid_results = get_best_results(validation_ret['results'], best_valid_results, epoch, model, optimizer, ckpt_dir, seed, save_best_model=False)
        
        test_ret = evaluate(model, dataLoader['test'], loss_fn, metrics_fn)

        best_test_results = get_best_results(test_ret['results'], best_test_results, epoch, model, optimizer, ckpt_dir, seed, save_best_model=True)

        end_time = time.time()
        epoch_mins, epoch_secs = interval_time(start_time, end_time)

        print(f'\n----------------- Results Epoch {epoch} -----------------')
        print(f'Learning Rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        print(f'Training Results: {training_ret["results"]}')
        if args.base.do_validation:
            print(f'Validation Results: {validation_ret["results"]}')
            print(f'Current Best Valid Results: {best_valid_results}')
        print(f'Test Results: {test_ret["results"]}\n')
        print(f'Current Best Test Results: {best_test_results}\n')
        print(f'Time: {epoch_mins}m {epoch_secs}s')
        print('----------------------------------------------------------\n')

        # writer.add_scalar('train/MAE', training_ret['loss_recorder'].value_avg, epoch)
        # writer.add_scalar('valid/MAE', validation_ret['loss_recorder'].value_avg, epoch)
        # writer.add_scalar('test/MAE', test_ret['loss_recorder'].value_avg, epoch)

        scheduler_warmup.step()

    # with open(f'./{args.dataset.datasetName}_results_all_epoch.txt', 'a+') as f:
    #     f.write(f'{seed}: {best_test_results["best_results_all_epochs"]}\n')
    
    # with open(f'./{args.dataset.datasetName}_results_one_epoch.txt', 'a+') as f:
    #     f.write(f'{seed}: {best_test_results["best_results_one_epoch"]}\n')

    # writer.close()


def train(model, data_loader, optimizer, loss_fn, metrics_fn):
    loss_recorder = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    for cur_iter, data in enumerate(data_loader):
        img, audio, text = data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        output = model(img, audio, text)

        loss = loss_fn(output, label)

        loss_recorder.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics_fn(pred, true)

    return {'results': results, 'loss_recorder': loss_recorder} 


def evaluate(model, data_loader, loss_fn, metrics_fn):
    loss_recorder = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    
    # 添加进度条
    data_loader = tqdm(data_loader, total=len(data_loader), desc="Evaluating")
    
    for cur_iter, data in enumerate(data_loader):
        # img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
        # Use masked data for evaluation to be consistent with LNLN/PRD-Net protocols
        img, audio, text = data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        with torch.no_grad():
            output = model(img, audio, text)

        loss = loss_fn(output, label)

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        loss_recorder.update(loss.item(), batchsize)

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics_fn(pred, true)

    return {'results': results, 'loss_recorder': loss_recorder} 

if __name__ == '__main__':
    main()
