# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IART (Implicit Alignment Restoration Transformer) is a video super-resolution project that enhances video quality via implicit resampling-based alignment. This is a PyTorch implementation of the CVPR 2024 Highlight paper.

## Key Commands

### Environment Setup
```bash
conda create -n IART python==3.9
conda activate IART
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

### Training
```bash
# Train with distributed training (8 GPUs)
bash dist_train.sh 8 options/IART_REDS_N6_300K.yml
bash dist_train.sh 8 options/IART_REDS_N16_600K.yml
bash dist_train.sh 8 options/IART_Vimeo_N14_300K.yml

# Custom training configurations
bash dist_train.sh 1 options/train_dvd_bluray.yml
```

Note: Training will terminate after 5000 iterations due to PyTorch checkpoint incompatibility with distributed training. Re-run the training script to resume.

### Testing
```bash
# Run demo
python demo.py

# Test REDS models
python test_scripts/BI/REDS/test_IART_REDS4_N6.py
python test_scripts/BI/REDS/test_IART_REDS4_N16.py
python test_scripts/BI/REDS/test_IART_Vid4_N6.py
python test_scripts/BI/REDS/test_IART_Vid4_N16.py

# Test Vimeo models
python test_scripts/BI/Vimeo-90K/test_IART_Vid4.py
python test_scripts/BI/Vimeo-90K/test_IART_Vimeo-90K-T.py
python test_scripts/BD/Vimeo-90K/test_IART_UDM10.py
```

## Architecture

### Core Components

**IART Model (`archs/iart_arch.py`)**
- Main architecture implementing the Implicit Alignment Restoration Transformer
- Key modules: ImplicitWarpModule for alignment, SwinIRFM for feature processing
- Supports multiple propagation branches (backward/forward)
- Configurable window sizes, depths, and attention heads

**Training Pipeline (`recurrent_mix_precision_train.py`)**
- Handles distributed training with mixed precision support
- Manages data loading, model initialization, and training loops
- Integrates with BasicSR framework for video super-resolution tasks

**Models (`models/recurrent_mixprecision_model.py`)**
- RecurrentMixPrecisionRTModel: Extends VideoRecurrentModel with mixed precision training
- Handles optimizer setup with separate learning rates for flow network
- Supports distributed data parallel training with static graph optimization

**Datasets (`data/`)**
- DVDRecurrentDataset: Handles DVD→Blu-ray restoration data
- VideoDataset: General video dataset implementation
- Supports multi-scale training with configurable patch sizes

### Configuration System

Training configurations are YAML-based (`options/`) with key parameters:
- `num_frame`: Number of frames per sequence (6, 14, or 16)
- `gt_size`: Patch size for training
- `embed_dim`, `depths`, `num_heads`: Architecture parameters
- `spynet_path`: Pre-trained optical flow network path

### Model Weights

Pre-trained models location: `experiments/pretrained_models/`
- IART_REDS_BI_N16.pth: REDS dataset, bicubic degradation, 16 frames
- SpyNet flow network: `flownet/spynet_sintel_final-3d2a1287.pth`

## Development Notes

### Key Dependencies
- PyTorch 1.13.1 with CUDA 11.7
- BasicSR framework for super-resolution utilities
- timm for vision transformers
- Optical flow via pre-trained SpyNet

### Data Preparation
Follow BasicSR dataset preparation guidelines. Expected structure:
```
datasets/
├── REDS/
│   ├── val_REDS4_sharp
│   └── val_REDS4_sharp_bicubic
```

### Training Considerations
- Mixed precision training with autocast for efficiency
- Separate learning rates for flow network (configurable via `flow_lr_mul`)
- Static graph optimization for distributed training
- EMA (Exponential Moving Average) for stable training