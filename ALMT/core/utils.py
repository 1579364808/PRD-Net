'''
* @name: utils.py
* @description: Other functions.
'''


import os
import random
import numpy as np
import torch
from types import SimpleNamespace


def interval_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class results_recorder(object):
    def __init__(self):
        self.best_results_one_epoch = {}
        # ['Has0_acc_2', 'Non0_acc_2', 'Mult_acc_2', 'Mult_acc_3', 'Mult_acc_5', 'Mult_acc_7', 'MAE', 'Corr']
        self.key_eval_one_epoch = 'MAE'
        self.best_results_all_epochs = {}
        self.epoch = None
        self.results = None

    def update(self, results, epoch):
        self.results = results
        self.epoch = epoch
        self.compute_best_results_one_epochs()
        self.compute_best_results_all_epochs()

    def get_best_results(self):
        return {'best_results_one_epoch': self.best_results_one_epoch, 'best_results_all_epochs': self.best_results_all_epochs}
    
    def compute_best_results_one_epochs(self):
        if self.epoch == 1:
            for key, value in self.results.items():
                self.best_results_one_epoch[key] = value

        elif self.results['MAE'] < self.best_results_one_epoch['MAE']:
            for key, value in self.results.items():
                self.best_results_one_epoch[key] = value

        else:
            pass

    def compute_best_results_all_epochs(self):
        if self.epoch == 1:
            for key, value in self.results.items():
                self.best_results_all_epochs[key] = value
        else:
            for key, value in self.results.items():
                if (key == 'Has0_acc_2') and (value > self.best_results_all_epochs[key]):
                    self.best_results_all_epochs[key] = value
                    self.best_results_all_epochs['Has0_F1_score'] = self.results['Has0_F1_score']

                elif (key == 'Non0_acc_2') and (value > self.best_results_all_epochs[key]):
                    self.best_results_all_epochs[key] = value
                    self.best_results_all_epochs['Non0_F1_score'] = self.results['Non0_F1_score']
                
                elif key == 'MAE' and value < self.best_results_all_epochs[key]:
                    self.best_results_all_epochs[key] = value
                    self.best_results_all_epochs['Corr'] = self.results['Corr']

                elif key == 'Mult_acc_2' and (value > self.best_results_all_epochs[key]):
                    self.best_results_all_epochs[key] = value
                    self.best_results_all_epochs['F1_score'] = self.results['F1_score']

                elif key == 'Mult_acc_3' or key == 'Mult_acc_5' or key == 'Mult_acc_7' or key == 'Corr':
                    if value > self.best_results_all_epochs[key]:
                        self.best_results_all_epochs[key] = value

                else:
                    pass
        

def save_model(save_path, epoch, model, optimizer):
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_path)


def get_best_results(results, best_results, epoch, model, optimizer, ckpt_root, seed, save_best_model):
    if epoch == 1:
        for key, value in results.items():
            best_results[key] = value
        
        if save_best_model:
            # Save initial best models for all relevant keys
            for key in ['Has0_acc_2', 'Non0_acc_2', 'MAE', 'Mult_acc_2', 'Mult_acc_3', 'Mult_acc_5', 'Mult_acc_7', 'Corr']:
                if key in results:
                    ckpt_path = os.path.join(ckpt_root, f'best_{key}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
    else:
        for key, value in results.items():
            if (key == 'Has0_acc_2') and (value > best_results[key]):
                best_results[key] = value
                best_results['Has0_F1_score'] = results['Has0_F1_score']

                if save_best_model:
                    key_eval = 'Has0_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
            
            elif (key == 'Non0_acc_2') and (value > best_results[key]):
                best_results[key] = value
                best_results['Non0_F1_score'] = results['Non0_F1_score']

                if save_best_model:
                    key_eval = 'Non0_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
            
            elif key == 'MAE' and value < best_results[key]:
                best_results[key] = value
                # best_results['Corr'] = results['Corr']

                if save_best_model:
                    key_eval = 'MAE'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)

            elif key == 'Mult_acc_2' and (value > best_results[key]):
                best_results[key] = value
                best_results['F1_score'] = results['F1_score']

                if save_best_model:
                    key_eval = 'Mult_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)

            elif key == 'Mult_acc_3' or key == 'Mult_acc_5' or key == 'Mult_acc_7' or key == 'Corr':
                if value > best_results[key]:
                    best_results[key] = value

                    if save_best_model:
                        key_eval = key
                        ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                        save_model(ckpt_path, epoch, model, optimizer)
            
            else:
                pass
    
    return best_results


def dict_to_namespace(d):
    """Recursively converts a dictionary and its nested dictionaries to a Namespace."""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)