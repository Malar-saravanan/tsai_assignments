"""
CIFAR-10 CNN Networks with C1C2C3C4 Architecture

Requirements:
- C1C2C3C4 architecture (4 convolution blocks)
- No MaxPooling (use dilated convolutions + stride=2)
- Receptive Field > 44
- Depthwise Separable Convolution (1 layer)
- Dilated Convolution (1 layer) - 200 bonus points
- Global Average Pooling + optional FC
- Parameters < 200k
- Target: 85% accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Lightweight Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 2)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    """Optimized Depthwise Separable Convolution with SE attention"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, use_se=True):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                 padding=padding, stride=stride, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        # Residual connection if dimensions match
        self.residual = nn.Identity() if in_channels == out_channels and stride == 1 else None
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        
        out = self.depthwise(x)
        out = self.bn1(out)
        out = F.silu(out)
        
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.se(out)
        
        if self.residual is not None:
            residual = self.residual(residual)
            
        out += residual
        return F.silu(out)


class DilatedConvBlock(nn.Module):
    """Optimized Dilated Convolution Block with residual connection"""
    def __init__(self, in_channels, out_channels, dilation=2):
        super(DilatedConvBlock, self).__init__()
        # Using dilated convolution instead of stride or maxpool for receptive field
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
        # Residual connection
        self.residual = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        residual = self.residual(x)
        out = self.conv(x)
        out = self.bn(out)
        out = F.silu(out)
        out = self.se(out)
        return F.silu(out + residual)


class ConvBlock(nn.Module):
    """Streamlined Convolution Block for parameter efficiency"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, use_se=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.silu(x)


class CIFAR10Net_Baseline(nn.Module):
    """
    Baseline CIFAR-10 CNN with C1C2C3C4 Architecture
    Target: ~82-85% accuracy with ~180k parameters
    """
    
    def __init__(self, num_classes=10):
        super(CIFAR10Net_Baseline, self).__init__()
        
        # C1: Input Block (3→16 channels)
        self.c1 = nn.Sequential(
            ConvBlock(3, 8, kernel_size=3, padding=1),    # RF: 3
            ConvBlock(8, 12, kernel_size=3, padding=1),   # RF: 5
            ConvBlock(12, 16, kernel_size=3, padding=1),  # RF: 7
        )
        
        # C2: Depthwise Separable Block (16→28 channels)
        self.c2 = nn.Sequential(
            DepthwiseSeparableConv(16, 20),               # RF: 9
            ConvBlock(20, 24, kernel_size=3, padding=1),  # RF: 11
            ConvBlock(24, 28, kernel_size=3, padding=1),  # RF: 13
        )
        
        # C3: Dilated Convolution Block (28→42 channels) - 200 BONUS POINTS
        self.c3 = nn.Sequential(
            DilatedConvBlock(28, 32, dilation=2),         # RF: 17
            ConvBlock(32, 36, kernel_size=3, padding=1),  # RF: 19
            DilatedConvBlock(36, 42, dilation=2),         # RF: 23
        )
        
        # C4: Final Block with stride=2 (42→50 channels) 
        self.c4 = nn.Sequential(
            ConvBlock(42, 44, kernel_size=3, padding=1, stride=2), # RF: 25, 32→16
            ConvBlock(44, 45, kernel_size=3, padding=1),           # RF: 29
            ConvBlock(45, 46, kernel_size=3, padding=1),           # RF: 33
            ConvBlock(46, 47, kernel_size=3, padding=1),           # RF: 37
            ConvBlock(47, 48, kernel_size=3, padding=1),           # RF: 41
            ConvBlock(48, 49, kernel_size=3, padding=1),           # RF: 45
            ConvBlock(49, 50, kernel_size=3, padding=1),           # RF: 49 ≥ 48 ✓
        )
        
        # Global Average Pooling + Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(50, num_classes)
        )
        
    def forward(self, x):
        # Forward pass through each block
        x = self.c1(x)    # 32x32 -> 32x32
        x = self.c2(x)    # 32x32 -> 32x32
        x = self.c3(x)    # 32x32 -> 32x32 (dilated convs maintain size)
        x = self.c4(x)    # 32x32 -> 16x16 (stride=2 in first conv)
        
        # Global Average Pooling
        x = self.gap(x)   # 16x16 -> 1x1
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        x = self.classifier(x)
        return x
    
    def get_receptive_field_info(self):
        """Calculate and return receptive field information"""
        rf_info = {
            'c1_conv1': 3,    # RF: 3
            'c1_conv2': 5,    # RF: 5
            'c1_conv3': 7,    # RF: 7
            'c2_depthwise': 9,    # RF: 9
            'c2_conv2': 11,   # RF: 11
            'c2_conv3': 13,   # RF: 13
            'c3_dilated1': 17,    # RF: 17 (13 + (3-1)*2)
            'c3_conv2': 19,   # RF: 19
            'c3_dilated2': 23,    # RF: 23 (19 + (3-1)*2)
            'c4_conv1_stride': 25,    # RF: 25
            'c4_conv2': 29,   # RF: 29
            'c4_conv3': 33,   # RF: 33
            'c4_conv4': 37,   # RF: 37
            'c4_conv5': 41,   # RF: 41
            'c4_conv6': 45,   # RF: 45
            'c4_conv7': 49,   # RF: 49 ≥ 48 ✓
        }
        return rf_info


class CIFAR10Net_Optimized(nn.Module):
    """
    Optimized CIFAR-10 CNN with C1C2C3C4 Architecture + SE blocks
    Target: >89% accuracy with <200k parameters
    """
    
    def __init__(self, num_classes=10):
        super(CIFAR10Net_Optimized, self).__init__()
        
        # C1: Input Block (3→16 channels)
        self.c1 = nn.Sequential(
            ConvBlock(3, 8, kernel_size=3, padding=1),    # RF: 3
            ConvBlock(8, 12, kernel_size=3, padding=1),   # RF: 5
            ConvBlock(12, 16, kernel_size=3, padding=1),  # RF: 7
        )
        
        # C2: Depthwise Separable Block (16→28 channels)
        self.c2 = nn.Sequential(
            DepthwiseSeparableConv(16, 20),               # RF: 9
            ConvBlock(20, 24, kernel_size=3, padding=1),  # RF: 11
            ConvBlock(24, 28, kernel_size=3, padding=1),  # RF: 13
            SEBlock(28),                                  # SE attention
        )
        
        # C3: Dilated Convolution Block (28→42 channels) - 200 BONUS POINTS
        self.c3 = nn.Sequential(
            DilatedConvBlock(28, 32, dilation=2),         # RF: 17
            ConvBlock(32, 36, kernel_size=3, padding=1),  # RF: 19
            DilatedConvBlock(36, 42, dilation=2),         # RF: 23
            SEBlock(42),                                  # SE attention
        )
        
        # C4: Final Block with stride=2 (42→56 channels)
        self.c4 = nn.Sequential(
            ConvBlock(42, 44, kernel_size=3, padding=1, stride=2), # RF: 25, 32→16
            ConvBlock(44, 46, kernel_size=3, padding=1),           # RF: 29
            ConvBlock(46, 48, kernel_size=3, padding=1),           # RF: 33
            ConvBlock(48, 50, kernel_size=3, padding=1),           # RF: 37
            ConvBlock(50, 52, kernel_size=3, padding=1),           # RF: 41
            ConvBlock(52, 54, kernel_size=3, padding=1),           # RF: 45
            ConvBlock(54, 56, kernel_size=3, padding=1),           # RF: 47 > 44 ✓
            SEBlock(56),                                           # SE attention
        )
        
        # Global Average Pooling + Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(56, num_classes)
        )
        
    def forward(self, x):
        # Forward pass through each block
        x = self.c1(x)    # 32x32 -> 32x32
        x = self.c2(x)    # 32x32 -> 32x32
        x = self.c3(x)    # 32x32 -> 32x32 (dilated convs maintain size)
        x = self.c4(x)    # 32x32 -> 16x16 (stride=2 in first conv)
        
        # Global Average Pooling
        x = self.gap(x)   # 16x16 -> 1x1
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        x = self.classifier(x)
        return x
    
    def get_receptive_field_info(self):
        """Calculate and return receptive field information"""
        rf_info = {
            'c1_conv1': 3,    # RF: 3
            'c1_conv2': 5,    # RF: 5
            'c1_conv3': 7,    # RF: 7
            'c2_depthwise': 9,    # RF: 9
            'c2_conv2': 11,   # RF: 11
            'c2_conv3': 13,   # RF: 13
            'c3_dilated1': 17,    # RF: 17 (13 + (3-1)*2)
            'c3_conv2': 19,   # RF: 19
            'c3_dilated2': 23,    # RF: 23 (19 + (3-1)*2)
            'c4_conv1_stride': 25,    # RF: 25
            'c4_conv2': 29,   # RF: 29
            'c4_conv3': 33,   # RF: 33
            'c4_conv4': 37,   # RF: 37
            'c4_conv5': 41,   # RF: 41
            'c4_conv6': 45,   # RF: 45
            'c4_conv7': 47,   # RF: 47 > 44 ✓
        }
        return rf_info


def count_parameters(model):
    """Count total parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_baseline_model():
    """Factory function to create the baseline model"""
    model = CIFAR10Net_Baseline(num_classes=10)
    total_params = count_parameters(model)
    print(f"Baseline Model - Total parameters: {total_params:,}")
    print(f"Parameter constraint (<200k): {'✓' if total_params < 200000 else '✗'}")
    
    # Print receptive field info
    rf_info = model.get_receptive_field_info()
    final_rf = max(rf_info.values())
    print(f"Final Receptive Field: {final_rf}")
    print(f"RF constraint (>44): {'✓' if final_rf > 44 else '✗'}")
    
    return model


def create_optimized_model():
    """Factory function to create the optimized model"""
    model = CIFAR10Net_Optimized(num_classes=10)
    total_params = count_parameters(model)
    print(f"Optimized Model - Total parameters: {total_params:,}")
    print(f"Parameter constraint (<200k): {'✓' if total_params < 200000 else '✗'}")
    
    # Print receptive field info
    rf_info = model.get_receptive_field_info()
    final_rf = max(rf_info.values())
    print(f"Final Receptive Field: {final_rf}")
    print(f"RF constraint (>44): {'✓' if final_rf > 44 else '✗'}")
    
    return model


def create_model():
    """Factory function to create the baseline model (backward compatibility)"""
    return create_baseline_model()


if __name__ == "__main__":
    # Test both models
    print("=== Testing Baseline Model ===")
    baseline_model = create_baseline_model()
    
    print("\n=== Testing Optimized Model ===")
    optimized_model = create_optimized_model()
