import pickle
import numpy as np
import random
import torch
import gc # 引入垃圾回收机制
from torch.utils.data import Dataset, DataLoader

__all__ = ['MMDataLoader', 'MMDataEvaluationLoader']

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.train_mode = args['base']['train_mode']
        self.datasetName = args['dataset']['datasetName']
        self.dataPath = args['dataset']['dataPath']
        self.missing_rate_eval_test = args['base']['missing_rate_eval_test']
        self.missing_seed = args['base']['seed']

        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        DATA_MAP[self.datasetName]()

    def __init_mosi(self):
        with open(self.dataPath, 'rb') as f:
            data = pickle.load(f)
        
        # --- 内存优化关键点 1: 不要保存 self.data = data ---
        # self.data = data  <-- 这一行删掉！绝对不能留！

        # 提取当前 mode 的数据
        self.text = data[self.mode]['text_bert'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)

        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        
        # 获取样本数量，用于后续逻辑，不再依赖 self.data
        self.num_samples = len(data[self.mode][self.train_mode+'_labels'])

        self.labels = {
            'M': data[self.mode][self.train_mode+'_labels'].astype(np.float32),
            'missing_rate_l': np.zeros(self.num_samples).astype(np.float32),
            'missing_rate_a': np.zeros(self.num_samples).astype(np.float32),
            'missing_rate_v': np.zeros(self.num_samples).astype(np.float32),
        }

        if self.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.train_mode+'_labels_'+m]

        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        # --- 内存优化关键点 2: 提取完所有需要的数据后，立刻删除 data 并回收内存 ---
        # 我们需要在这里先处理完 labels/missing_rate 的初始化逻辑，再删除 data 也可以，
        # 但为了保险，我们先记录所有需要的信息，上面的 labels 已经提取完了。
        
        # 初始化 Missing Rate 逻辑
        if self.mode == 'train':
            missing_rate = [np.random.uniform(size=(self.num_samples, 1)) for i in range(3)]
            
            for i in range(3):
                sample_idx = random.sample([k for k in range(len(missing_rate[i]))], int(len(missing_rate[i])/2))
                missing_rate[i][sample_idx] = 0

            self.labels['missing_rate_l'] = missing_rate[0].reshape(-1) # 确保维度匹配
            self.labels['missing_rate_a'] = missing_rate[1].reshape(-1)
            self.labels['missing_rate_v'] = missing_rate[2].reshape(-1)
        else:
            missing_rate = [self.missing_rate_eval_test * np.ones((self.num_samples, 1)) for i in range(3)]
            self.labels['missing_rate_l'] = missing_rate[0].reshape(-1)
            self.labels['missing_rate_a'] = missing_rate[1].reshape(-1)
            self.labels['missing_rate_v'] = missing_rate[2].reshape(-1)   

        # 生成 Mask 结构
        self.text_mask_struct = self.generate_mask_only(self.text[:,0,:], self.text[:,1,:], None,
                                                        missing_rate[0], self.missing_seed, mode='text')

        self.audio_mask_struct = self.generate_mask_only(self.audio, None, self.audio_lengths,
                                                         missing_rate[1], self.missing_seed, mode='audio')
        self.vision_mask_struct = self.generate_mask_only(self.vision, None, self.vision_lengths,
                                                          missing_rate[2], self.missing_seed, mode='vision')
        
        # --- 彻底清理内存 ---
        del data
        gc.collect()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def generate_mask_only(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):
        if mode == 'text':
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
        
        np.random.seed(missing_seed)
        # 确保 missing_rate 形状正确 (N, 1)
        if missing_rate.ndim == 1:
            missing_rate = missing_rate.reshape(-1, 1)

        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate.repeat(input_mask.shape[1], 1)) * input_mask
        
        assert missing_mask.shape == input_mask.shape
        
        if mode == 'text':
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1
            
        return {'missing_mask': missing_mask, 'input_len': input_len, 'input_mask': input_mask}

    def apply_mask_single(self, index, modality_type):
        if modality_type == 'text':
            mask_info = self.text_mask_struct
            
            raw_input_ids = self.text[index, 0, :]
            raw_input_mask = self.text[index, 1, :]
            raw_segment_ids = self.text[index, 2, :]

            missing_mask = mask_info['missing_mask'][index] 
            input_mask = mask_info['input_mask'][index]

            masked_input_ids = missing_mask * raw_input_ids + (100.0) * (input_mask - missing_mask)
            
            return np.stack([masked_input_ids, raw_input_mask, raw_segment_ids], axis=0)

        elif modality_type == 'audio':
            mask_info = self.audio_mask_struct
            raw_data = self.audio[index]
            missing_mask = mask_info['missing_mask'][index]
            return missing_mask.reshape(-1, 1) * raw_data

        elif modality_type == 'vision':
            mask_info = self.vision_mask_struct
            raw_data = self.vision[index]
            missing_mask = mask_info['missing_mask'][index]
            return missing_mask.reshape(-1, 1) * raw_data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 保持原来的 index == 0 逻辑，但不再依赖 self.data
        if (self.mode == 'train') and (index == 0):
            # 使用 self.num_samples 替代 len(self.data[...])
            missing_rate = [np.random.uniform(size=(self.num_samples, 1)) for i in range(3)]
            
            for i in range(3):
                sample_idx = random.sample([k for k in range(len(missing_rate[i]))], int(len(missing_rate[i])/2))
                missing_rate[i][sample_idx] = 0

            # 更新 labels 中的 missing rate
            self.labels['missing_rate_l'] = missing_rate[0].reshape(-1)
            self.labels['missing_rate_a'] = missing_rate[1].reshape(-1)
            self.labels['missing_rate_v'] = missing_rate[2].reshape(-1)

            # 重新生成 Mask
            self.text_mask_struct = self.generate_mask_only(self.text[:,0,:], self.text[:,1,:], None,
                                                            missing_rate[0], self.missing_seed, mode='text')
            self.audio_mask_struct = self.generate_mask_only(self.audio, None, self.audio_lengths,
                                                             missing_rate[1], self.missing_seed, mode='audio')
            self.vision_mask_struct = self.generate_mask_only(self.vision, None, self.vision_lengths,
                                                              missing_rate[2], self.missing_seed, mode='vision')

        text_m_sample = self.apply_mask_single(index, 'text')
        audio_m_sample = self.apply_mask_single(index, 'audio')
        vision_m_sample = self.apply_mask_single(index, 'vision')

        sample = {
            'text': torch.Tensor(self.text[index]),
            'text_m': torch.Tensor(text_m_sample), 
            'audio': torch.Tensor(self.audio[index]),
            'audio_m': torch.Tensor(audio_m_sample),
            'vision': torch.Tensor(self.vision[index]),
            'vision_m': torch.Tensor(vision_m_sample),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }

        return sample


def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['base']['batch_size'],
                       num_workers=args['base']['num_workers'],
                       shuffle=True if ds == 'train' else False)
        for ds in datasets.keys()
    }
    
    return dataLoader

def MMDataEvaluationLoader(args):
    datasets = MMDataset(args, mode='test')

    dataLoader = DataLoader(datasets,
                       batch_size=args['base']['batch_size'],
                       num_workers=args['base']['num_workers'],
                       shuffle=False)
    
    return dataLoader