import os
import sys
import torch
import yaml
import argparse
import copy
from core.dataset import MMDataEvaluationLoader
from core.metric import MetricsTop
from models.almt import build_model
from core.utils import dict_to_namespace


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)


def set_nested_value(d, key_path, value):
    """设置嵌套字典的值，例如 'model.prm.codebook_size' -> d['model']['prm']['codebook_size']"""
    keys = key_path.split('.')
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]

    # 尝试转换类型
    try:
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
    except ValueError:
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False

    d[keys[-1]] = value


def parse_override_args(args_list):
    """解析 --param.path=value 格式的参数覆盖"""
    overrides = {}
    for arg in args_list:
        if arg.startswith('--') and '=' in arg:
            key_value = arg[2:]  # 去掉 '--'
            key, value = key_value.split('=', 1)
            overrides[key] = value
    return overrides


# 分离已知参数和覆盖参数
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='')
parser.add_argument('--key_eval', type=str, default='')
parser.add_argument('--ckpt_dir', type=str, default=None, help='自定义checkpoint目录')
opt, remaining_args = parser.parse_known_args()

# 解析覆盖参数
override_args = parse_override_args(remaining_args)
print(f"Config overrides: {override_args}")
print(opt)


def main():
    config_file = 'configs/sims.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)

    # 应用命令行参数覆盖（用于网格搜索时保持与训练时相同的参数）
    for key_path, value in override_args.items():
        set_nested_value(args_dict, key_path, value)
        print(f"Override: {key_path} = {value}")

    # Convert to namespace for model building
    args = dict_to_namespace(copy.deepcopy(args_dict))
    print(args)

    dataset_name = args.dataset.datasetName
    # 如果配置文件中没有key_eval，则使用命令行参数，否则使用配置文件中的值
    key_eval = opt.key_eval

    print("Using ALMT model")
    model = build_model(args).to(device)
    metrics = MetricsTop().getMetics(dataset_name)

    missing_rate_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for cur_r in missing_rate_list:
        test_results_list = []
        if dataset_name == 'sims':
            for _, cur_seed  in enumerate([1111, 1112, 1113]):
                if opt.ckpt_dir:
                    best_ckpt = os.path.join(opt.ckpt_dir, f'seed_{cur_seed}', f'best_{key_eval}_{cur_seed}.pth')
                else:
                    best_ckpt = os.path.join(f'ckpt/{dataset_name}/best_{key_eval}_{cur_seed}.pth')
                
                if not os.path.exists(best_ckpt):
                    print(f"Warning: Checkpoint not found: {best_ckpt}")
                    continue

                model.load_state_dict(torch.load(best_ckpt, weights_only=False)['state_dict'])
                
                # Update missing rate in args
                args.base.missing_rate_eval_test = cur_r 
                args_dict['base']['missing_rate_eval_test'] = cur_r

                dataLoader = MMDataEvaluationLoader(args_dict)
        
                test_results_cur_seed = evaluate(model, dataLoader, metrics)
                print(f'Seed {cur_seed} - key_eval: {key_eval}, missing rate: {cur_r}, results: {test_results_cur_seed}')
                
                test_results_list.append(test_results_cur_seed)

            if key_eval == 'Mult_acc_2':
                Mult_acc_2_avg = (test_results_list[0]['Mult_acc_2'] + test_results_list[1]['Mult_acc_2'] + test_results_list[2]['Mult_acc_2']) / 3
                F1_score_avg = (test_results_list[0]['F1_score'] + test_results_list[1]['F1_score'] + test_results_list[2]['F1_score']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_2_avg: {Mult_acc_2_avg}, F1_score_avg: {F1_score_avg}')
            elif key_eval == 'Mult_acc_3':
                Mult_acc_3_avg = (test_results_list[0]['Mult_acc_3'] + test_results_list[1]['Mult_acc_3'] + test_results_list[2]['Mult_acc_3']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_3_avg: {Mult_acc_3_avg}')
            elif key_eval == 'Mult_acc_5':
                Mult_acc_5_avg = (test_results_list[0]['Mult_acc_5'] + test_results_list[1]['Mult_acc_5'] + test_results_list[2]['Mult_acc_5']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_5_avg: {Mult_acc_5_avg}')
            elif key_eval == 'MAE':
                MAE_avg = (test_results_list[0]['MAE'] + test_results_list[1]['MAE'] + test_results_list[2]['MAE']) / 3
                Corr_avg = (test_results_list[0]['Corr'] + test_results_list[1]['Corr'] + test_results_list[2]['Corr']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, MAE_avg: {MAE_avg}, Corr_avg: {Corr_avg}')
            
        else:
            for _, cur_seed  in enumerate([1111, 1112, 1113]):
                if opt.ckpt_dir:
                    best_ckpt = os.path.join(opt.ckpt_dir, f'seed_{cur_seed}', f'best_{key_eval}_{cur_seed}.pth')
                else:
                    best_ckpt = os.path.join(f'ckpt/{dataset_name}/best_{key_eval}_{cur_seed}.pth')
                
                if not os.path.exists(best_ckpt):
                    print(f"Warning: Checkpoint not found: {best_ckpt}")
                    continue

                model.load_state_dict(torch.load(best_ckpt, weights_only=False)['state_dict'])
                args.base.missing_rate_eval_test = cur_r 
                args_dict['base']['missing_rate_eval_test'] = cur_r

                dataLoader = MMDataEvaluationLoader(args_dict)
        
                test_results_cur_seed = evaluate(model, dataLoader, metrics)
                print(f'Seed {cur_seed} - key_eval: {key_eval}, missing rate: {cur_r}, results: {test_results_cur_seed}')
                
                test_results_list.append(test_results_cur_seed)

            if key_eval == 'Has0_acc_2':
                Has0_acc_2_avg = (test_results_list[0]['Has0_acc_2'] + test_results_list[1]['Has0_acc_2'] + test_results_list[2]['Has0_acc_2']) / 3
                Has0_F1_score_avg = (test_results_list[0]['Has0_F1_score'] + test_results_list[1]['Has0_F1_score'] + test_results_list[2]['Has0_F1_score']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Has0_acc_2_avg: {Has0_acc_2_avg}, F1_score_avg: {Has0_F1_score_avg}')
            elif key_eval == 'Non0_acc_2':
                Non0_acc_2_avg = (test_results_list[0]['Non0_acc_2'] + test_results_list[1]['Non0_acc_2'] + test_results_list[2]['Non0_acc_2']) / 3
                Non0_F1_score_avg = (test_results_list[0]['Non0_F1_score'] + test_results_list[1]['Non0_F1_score'] + test_results_list[2]['Non0_F1_score']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Non0_acc_2_avg: {Non0_acc_2_avg}, Non0_F1_score_avg: {Non0_F1_score_avg}')
            elif key_eval == 'Mult_acc_5':
                Mult_acc_5_avg = (test_results_list[0]['Mult_acc_5'] + test_results_list[1]['Mult_acc_5'] + test_results_list[2]['Mult_acc_5']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_5_avg: {Mult_acc_5_avg}')
            elif key_eval == 'Mult_acc_7':
                Mult_acc_7_avg = (test_results_list[0]['Mult_acc_7'] + test_results_list[1]['Mult_acc_7'] + test_results_list[2]['Mult_acc_7']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_7_avg: {Mult_acc_7_avg}')
            elif key_eval == 'MAE':
                MAE_avg = (test_results_list[0]['MAE'] + test_results_list[1]['MAE'] + test_results_list[2]['MAE']) / 3
                Corr_avg = (test_results_list[0]['Corr'] + test_results_list[1]['Corr'] + test_results_list[2]['Corr']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, MAE_avg: {MAE_avg}, Corr_avg: {Corr_avg}')


def evaluate(model, eval_loader, metrics):
    y_pred, y_true = [], []

    model.eval()
    for cur_iter, data in enumerate(eval_loader):
        # ALMT model expects: img, audio, text
        # MMDataEvaluationLoader returns: text_m, audio_m, vision_m (masked)
        # We should use the masked versions for robust evaluation
        
        img = data['vision_m'].to(device)
        audio = data['audio_m'].to(device)
        text = data['text_m'].to(device)
        
        sentiment_labels = data['labels']['M'].to(device)
        
        with torch.no_grad():
            out = model(img, audio, text)

        y_pred.append(out.cpu())
        y_true.append(sentiment_labels.cpu())
    
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    return results


if __name__ == '__main__':
    main()
