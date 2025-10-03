# CIFAR-10 CNN Assignment

## Requirements
- Python 3.8+
- PyTorch >= 1.12
- torchvision
- albumentations
- numpy, matplotlib

Install:
```bash
pip install -r requirements.txt
```

## Architecture Implementation

### C1C2C3C4 Structure
- **C1**: 3 ConvBlocks (3→16 channels)
- **C2**: DepthwiseSeparableConv + 2 ConvBlocks + SEBlock (16→28 channels)
- **C3**: 2 DilatedConvBlocks + SEBlock (28→42 channels, dilation=2)
- **C4**: 7 ConvBlocks + SEBlock with stride=2 (42→56 channels)

### Key Features
- **No MaxPooling**: Uses dilated convolutions and stride=2 for downsampling
- **Depthwise Separable Conv**: Implemented in C2 block with groups=in_channels
- **Dilated Convolution**: Implemented in C3 block with dilation=2
- **Global Average Pooling**: nn.AdaptiveAvgPool2d(1) followed by FC layer
- **Receptive Field**: 47
- **Parameters**: 177,379

## Data Augmentation
Implemented using Albumentations library with exact specifications:

```python
A.HorizontalFlip(p=0.5)
A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
A.CoarseDropout(
    num_holes_range=(1,1),      # max_holes=1, min_holes=1
    hole_height_range=(16,16),  # max_height=16px, min_height=16px
    hole_width_range=(16,16),   # max_width=16px, min_width=16px
    fill=dataset_mean,          # fill_value=dataset_mean
    fill_mask=None,             # mask_fill_value=None
    p=0.5
)
A.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
```

## Dataset Information
- **Training Set**: 50,000 samples
- **Test Set**: 10,000 samples  
- **Validation Set**: 8,000 samples (16% of training data)
- **Final Training Set**: 42,000 samples (after validation split)

## Experiments

### Experiment 1: Baseline Model

**Target:**
- Achieve 85% test accuracy
- Parameters < 200K
- Validate C1C2C3C4 architecture

**Results:**
- **Test Accuracy**: 86.32%
- **Validation Accuracy**: 86.84%
- **Parameters**: 186,770
- **Training Time**: 29.9 minutes (20 epochs)
- **Device**: MPS (Apple Silicon)

**Analysis:**
- Successfully exceeded the 85% accuracy target with baseline architecture
- Efficient parameter usage (186K < 200K limit)
- Early stopping at epoch 20 due to target achievement
- Demonstrates effectiveness of C1C2C3C4 structure without max pooling

### Experiment 2: Optimized Model

**Target:**
- Achieve 89% test accuracy
- Further optimize parameters and performance
- Incorporate advanced techniques (SE blocks, better regularization)

**Results:**
- **Test Accuracy**: 89.11%
- **Validation Accuracy**: 89.36%
- **Parameters**: 177,379
- **Training Time**: 43.7 minutes (29 epochs)
- **Device**: MPS (Apple Silicon)

**Analysis:**
- Successfully achieved 89% accuracy target
- **Performance Improvement**: +2.79% over baseline (86.32% → 89.11%)
- **Parameter Efficiency**: Reduced parameters by 9,391 while improving accuracy
- **Key Optimizations**: SE attention blocks, improved regularization (dropout 0.3), optimized channel progression
- Training converged at epoch 27-29 with consistent high performance

### Network Structure
**C1C2C3C4 Architecture** with 4 distinct convolution blocks:
- **C1 Block**: 3 ConvBlocks (3→16 channels) - Input processing
- **C2 Block**: DepthwiseSeparableConv + 2 ConvBlocks + SEBlock (16→28 channels) - Feature extraction
- **C3 Block**: 2 DilatedConvBlocks + SEBlock (28→42 channels, dilation=2) - Receptive field expansion
- **C4 Block**: 7 ConvBlocks + SEBlock with stride=2 (42→56 channels) - High-level features

### Advanced Techniques
- **No MaxPooling**: Uses dilated convolutions and stride=2 for downsampling
- **Depthwise Separable Convolution**: Reduces parameters while maintaining performance
- **Dilated Convolution**: Increases receptive field without spatial resolution loss
- **Squeeze-and-Excitation (SE) Blocks**: Channel attention mechanism for better feature selection
- **Global Average Pooling**: Reduces overfitting and parameter count
- **Residual Connections**: Improves training stability and gradient flow

### Model Specifications
- **Input Shape**: (3, 32, 32) - CIFAR-10 images
- **Output Classes**: 10 (CIFAR-10 categories)
- **Receptive Field**: 47 (calculated through all layers)
- **Architecture Depth**: 4 major blocks with progressive channel expansion
- **Memory Efficiency**: Optimized through depthwise separable convolutions and GAP

## Implementation Checklist

| Requirement | Specification | Implementation | Value/Status |
|-------------|---------------|----------------|--------------|
| **Dataset** | CIFAR-10 | `torchvision.datasets.CIFAR10` | 50K train, 10K test |
| **Architecture** | C1C2C3C4 | 4 convolution blocks | C1(3→16), C2(16→28), C3(28→42), C4(42→56) |
| **MaxPooling** | Not allowed | Dilated conv + stride=2 | No MaxPool2d used |
| **Depthwise Separable** | Required | In C2 block | `groups=in_channels` |
| **Dilated Convolution** | Required | In C3 block | `dilation=2` |
| **Receptive Field** | > 44 | Calculated | 47 |
| **Parameters** | < 200K | Model size | 177,379 |
| **Accuracy Target** | ≥ 85% | Test accuracy | 89.11% |
| **GAP + FC** | Required | After C4 block | `AdaptiveAvgPool2d(1)` + `Linear` |
| **Augmentations** | 3 specific | Albumentations | HorizontalFlip, ShiftScaleRotate, CoarseDropout |
| **Code Structure** | Modular | Separated modules | `models/`, `data/`, `experiments/` |
| **Training Logs** | Required | Complete logging | Train/Val/Test per epoch |

## Usage
```bash
python run_complete_experiments.py
```

Results stored in `logs/` and `results/` folders.
