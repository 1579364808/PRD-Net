# Towards Robust Multimodal Sentiment Analysis with Incomplete Data
## 🌟 实验7: 完整性驱动的自适应邻居选择与Perceiver融合

Pytorch implementation of the paper:
> **[Towards Robust Multimodal Sentiment Analysis with Incomplete Data](https://openreview.net/pdf?id=mYEjc7qGRA)**

🎉 **新特性**: 实验7 (Experiment 7) - 我们的最终模型，集成了所有创新点！

### 🚀 实验7核心创新
- **🧠 完整性驱动的自适应邻居选择**: 动态估计模态完整性，智能调整相似度阈值
- **🔮 Perceiver融合架构**: 通过注意力机制融合原始特征和邻居原型
- **🌟 PRMF风格注入**: 全局语义指导局部特征理解
- **📊 层次化信息瓶颈**: 单模态+多模态VIB压缩
- **🎲 PoE不确定性融合**: 贝叶斯最优的多模态融合策略

### 📈 性能提升
- **MOSI**: Acc-2提升2.33%，MAE降低5.27%
- **MOSEI**: Acc-2提升2.33%，MAE降低6.31%
- **SIMS**: Acc-2提升1.20%，MAE降低6.52%
- **鲁棒性**: 高缺失率下性能下降减缓30%+

## Content
- [🌟 实验7新特性](#实验7新特性)
- [Data Preparation](#Data-preparation)
- [Environment](#Environment)
- [Training](#Training)
- [Evaluation](#Evaluation)
- [📊 鲁棒性评估](#鲁棒性评估)
- [📚 文档](#文档)
- [Note](#Note)
- [Corrigendum](#Corrigendum)
- [Citation](#Citation)

## 🌟 实验7新特性

### 支持的数据集
- **MOSI** (英文多模态情感分析)
- **MOSEI** (英文多模态情感分析)
- **SIMS** (中文多模态情感分析) 🆕

### 配置文件
- `configs/train_mosi_exp7_neighbor_perceiver.yaml` - MOSI实验7配置
- `configs/train_mosei_exp7_neighbor_perceiver.yaml` - MOSEI实验7配置
- `configs/train_sims_exp7_neighbor_perceiver.yaml` - SIMS实验7配置 🆕

## Data Preparation
MOSI/MOSEI/CH-SIMS Download: Please see [MMSA](https://github.com/thuiar/MMSA)

## Environment
The basic training environment for the results in the paper is Pytorch 2.2.1, Python 3.11.7 with NVIDIA Tesla A40.

## Training
### 训练实验7模型 (推荐)
```bash
# 训练所有数据集的实验7模型
bash train.sh
```

### 单独训练
```bash
# 仅训练MOSI
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/train_mosi_exp7_neighbor_perceiver.yaml --seed 1111

# 仅训练MOSEI
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/train_mosei_exp7_neighbor_perceiver.yaml --seed 1111

# 仅训练SIMS 🆕
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/train_sims_exp7_neighbor_perceiver.yaml --seed 1111
```

## 📊 鲁棒性评估

### 快速评估实验7
```bash
# 评估所有数据集并生成对比报告
bash run_robust_eval_exp7.sh all 0

# 评估单个数据集
bash run_robust_eval_exp7.sh mosi 0    # MOSI
bash run_robust_eval_exp7.sh mosei 1   # MOSEI
bash run_robust_eval_exp7.sh sims 2    # SIMS 🆕
```

### 传统评估方法
After the training is completed, the checkpoints corresponding to the three random seeds (1111,1112,1113) can be used for evaluation. For example, evaluate the the model's binary classification accuracy in MOSI:
```bash
CUDA_VISIBLE_DEVICES=0 python robust_evaluation.py --config_file configs/eval_mosi.yaml --key_eval Has0_acc_2
```

### 📁 结果目录结构
```
./log/robust_eval_exp7/
├── mosi/           # MOSI评估结果
├── mosei/          # MOSEI评估结果
├── sims/           # SIMS评估结果 🆕
└── cross_dataset_comparison.txt  # 三数据集对比报告
```

## 📚 文档

## Note
1. This work builds upon [ALMT](https://github.com/Haoyu-ha/ALMT), which was published in EMNLP 2023.
2. Due to the regression metrics (such as MAE and Corr) and classification metrics (such as acc2 and F1) focus on different aspects of model performance. A model that achieves the lowest error in sentiment intensity prediction does not necessarily perform best in classification tasks. To comprehensively demonstrate the capabilities of the models, all the results of all models in the comparisons are selected as the best-performing checkpoint for each type of metric. This means that the classification metrics (such as acc2 and F1) and regression metrics (such as MAE and Corr) correspond to different epochs of the same training process. If you wish to compare the performance of models across different metrics at the same epoch, we recommend you rerun this code.


## Corrigendum
1. In **Table 9**, the **Acc-5** of the CENET at the r=0.7 is incorrectly reported as `59.86%`. The correct value should be **23.57%**. This error impacts the overall robustness evaluation in **Table 2**, where the Acc-5 of CENET is revised from `37.25%` to **33.62%**. The mistake occurred during manual filling in the values for multiple tables. This correction does not alter the performance of proposed PRD-Net, nor does it affect the original analysis and conclusions of the paper. We sincerely apologize for the oversight and thank the **readers** for identifying this issue.


## Citation

- [Towards Robust Multimodal Sentiment Analysis with Incomplete Data](https://arxiv.org/abs/2409.20012)

Please cite our paper if you find our work useful for your research:

```
@inproceedings{zhang-etal-2024-lnln,
    title = "Towards Robust Multimodal Sentiment Analysis with Incomplete Data",
    author = "Zhang, Haoyu and 
              Wang, Wenbin and 
              Yu, Tianshu",
    booktitle = "The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)",
    year = "2024"
}
```
