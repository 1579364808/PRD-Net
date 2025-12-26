# PRD-Net: Prototype Rectification and Denoising Network for Robust Multimodal Sentiment Analysis with Incomplete Data

This is the official repository for the paper: **PRD-Net: Prototype Rectification and Denoising Network for Robust Multimodal Sentiment Analysis with Incomplete Data**.

## Projects

### Our Model
- **[PRD-Net](PRD-Net/)**: PRD-Net: Prototype Rectification and Denoising Network for Robust Multimodal Sentiment Analysis with Incomplete Data

### Reproduced Models
We provide reproductions of the following state-of-the-art models. Please refer to their original repositories for more details:

- **[ALMT](ALMT/)**: Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis
  - Original Repo: [https://github.com/Haoyu-ha/ALMT](https://github.com/Haoyu-ha/ALMT)
- **[LNLN](LNLN/)**: Towards Robust Multimodal Sentiment Analysis with Incomplete Data
  - Original Repo: [https://github.com/Haoyu-ha/LNLN](https://github.com/Haoyu-ha/LNLN)
- **[P-RMF](P-RMF/)**: Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data
  - Original Repo: [https://github.com/aoqzhu/P-RMF](https://github.com/aoqzhu/P-RMF)

## Setup

1. Environment Setup:
   ```bash
   conda create -n Pytorch python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 scikit-learn einops transformers matplotlib seaborn -c pytorch -c nvidia -c conda-forge
   conda activate Pytorch
   ```

2. Prepare Datasets:
   MOSI/MOSEI/CH-SIMS Download: Please see [MMSA](https://github.com/thuiar/MMSA)

   Place your datasets in a `datasets` folder in the root or symlink it.
   Structure:
   ```
   datasets/
   ├── MOSI/
   │   └── Processed/
   │       └── unaligned_50.pkl
   ├── MOSEI/
   │   └── Processed/
   │       └── unaligned_50.pkl
   └── CH-SIMS/
       └── Processed/
           └── unaligned_39.pkl
   ```

## Training and Evaluation

Navigate to each project folder (e.g., `cd PRD-Net`) to run experiments.

**Training:**
You can use the provided shell script or run the python script directly:
```bash
# Method 1: Using shell script (Recommended)
bash train.sh

# Method 2: Using python
python train.py --config_file configs/train_mosi.yaml
```

**Evaluation:**
Use `robust_evaluation.py` for evaluation:
```bash
python robust_evaluation.py --config_file configs/eval_mosi.yaml --key_eval Non0_acc_2
```
