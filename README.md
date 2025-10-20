# FCD Detection Project

Deep learning-based Focal Cortical Dysplasia (FCD) detection in MRI scans using PyTorch and MONAI.

## Features

- Multi-scale network architectures
- Mixed precision training
- Comprehensive evaluation metrics
- WandB integration for experiment tracking
- Flexible data augmentation pipeline
- Automatic model checkpointing
- Support for resuming training

## Requirements

- Python 3.12+
- PyTorch 2.6+ (cuda 12.6+)
- MONAI 1.5+
- Weights & Biases (wandb)
- CUDA-capable GPU

## Installation

Clone this git repository

Create a python environment

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with your WandB API key:

```bash
WANDB_API_KEY=your_api_key_here
```

## Usage

### Training

```bash
python train.py \
    --data_dir /path/to/dataset \
    --split_file /path/to/split_file \
    --save_dir /path/to/save/models \
    --model_type MS_DSA_NET \
    --kwargs loss=DiceCELoss batch_size=1
```

### Testing

```bash
python train.py \
    --data_dir /path/to/dataset \
    --split_file /path/to/split_file \
    --splits test
    --model_type MS_DSA_NET \
    --checkpoint_path /path/to/model/checkpoint.pth \
```

### Additional Parameters

- `--resume`: Resume training from the latest checkpoint
- `--prefix`: Add a prefix to the model save directory
- `--kwargs`: Override default parameters. Common options include:
  - `lr`: Learning rate (default: 1e-4)
  - `batch_size`: Batch size (default: 1)
  - `loss`: Loss function ['DiceLoss', 'DiceCELoss', 'DiceFocalLoss']
  - `use_amp`: Enable mixed precision training (default: True)
  - `deterministic`: Set random seed mode ['off', 'seed_only', 'strict']

## Model Types

- MS_DSA_NET, MS_DSA_NET_PS: Multi-scale Dual Self-Attention Network with configurable pixelshuffle upsampling
- SEGRESNET, SEGRESNETVAE: Residual Encoder-Decoder Network with optional Variational Auto-Encoder
- SEGRESNET_DSA: combined network of SegResNet and MS_DSA_NET
- BaseUNet : Standard 3D U-Net architecture similar to MS_DSA_NET but with standard skip connections

## Features Details

### Training

- Automatic mixed precision training
- Layer-wise learning rate decay (LLRD)
- Early stopping with EMA validation loss
- Learning rate scheduling (warmup + cosine annealing)
- Total Variation loss option
- Automatic post-processing
- Checkpoint saving (best and latest models)

### Evaluation Metrics

- Dice coefficient
- Hausdorff distance (95th percentile)
- Average surface distance
- Precision, Recall, F1-score
- Subject-level metrics

### Experiment Tracking

- WandB integration for metrics logging
- CSV logging for offline analysis
- Training configuration tracking
- Validation metrics monitoring


## Acknowledgments

This project builds upon the following works:
- [MS-DSA-NET](https://github.com/zhangxd0530/MS-DSA-NET) for the multi-scale dual self-attention network architecture
- [BraTS-2023-Metrics](https://github.com/rachitsaluja/BraTS-2023-Metrics) for evaluation metrics implementation

## Citation

If you use this code in your research, please cite:

```bibtex
Rabiee, M., Greco, S., Shahbazian, R. & Trubitsyna, I. A total variation regularized framework for epilepsy-related mri
image segmentation. arXiv preprint arXiv:2510.06276 (2025). 2510.06276.
```


