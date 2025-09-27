"""
ITERATIVE MODEL DEVELOPMENT FOR MNIST CNN
Following the pattern from notes.txt for systematic improvement

Target: Achieve 99.4% validation accuracy consistently in the last few epochs 
with ≤8000 parameters in ≤15 epochs using iterative learning approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Model_1(nn.Module):
    """
    Model_1: Basic Skeleton Setup
    
    Target:
    Get the basic CNN skeleton right with minimal parameters. Establish baseline 
    architecture with conv layers, batch norm, and basic structure. No regularization 
    or fancy techniques - just core CNN building blocks.
    
    Results:
    Parameters: 7,450
    Best Training Accuracy: 99.08%
    Best Test Accuracy: 99.04%
    Epochs to converge: 15
    
    Analysis:
    The basic model shows good learning capacity with minimal overfitting (train: 99.08% 
    vs test: 99.04%). The small gap indicates decent generalization, but the test 
    accuracy of 99.04% is still short of our 99.4% target. The architecture has 
    sufficient capacity, so we need regularization to improve generalization further.
    """
    def __init__(self):
        super(Model_1, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # RF=3, 28x28
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1)  # RF=5, 28x28 
        self.bn2 = nn.BatchNorm2d(12)
        self.pool1 = nn.MaxPool2d(2, 2)  # RF=6, 14x14 (MaxPool at RF=5 as per notes)
        
        self.conv3 = nn.Conv2d(12, 14, 3, padding=1)  # RF=10, 14x14
        self.bn3 = nn.BatchNorm2d(14)
        self.pool2 = nn.MaxPool2d(2, 2)  # RF=12, 7x7
        
        self.conv4 = nn.Conv2d(14, 16, 3)  # RF=20, 5x5 (no padding for capacity)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 18, 3)  # RF=24, 3x3 (additional conv for capacity)
        self.bn5 = nn.BatchNorm2d(18)
        
        # GAP + 1x1 conv for classification (as per notes.txt)
        self.conv6 = nn.Conv2d(18, 10, 1)  # RF=24, 3x3
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

class Model_2(nn.Module):
    """
    Model_2: Add Regularization
    
    Target:
    Reduce overfitting by adding dropout after pooling layers. Add data augmentation 
    (rotation) to make training harder and improve generalization. Keep same 
    architecture but add regularization techniques.
    
    Results:
    Parameters: 7,450 (same as Model_1)
    Best Training Accuracy: 98.51%
    Best Test Accuracy: 99.21%
    Epochs to converge: 15
    
    Analysis:
    Regularization is working! The model now generalizes better with test accuracy 
    (99.21%) higher than training (98.51%), showing the dropout and augmentation 
    are preventing overfitting. The +0.17% improvement over Model_1 demonstrates 
    that regularization helps. The model is slightly under-fitting, which provides 
    room for training optimization.
    """
    def __init__(self):
        super(Model_2, self).__init__()
        # Same architecture as Model_1
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.1)  # Add dropout after pooling
        
        self.conv3 = nn.Conv2d(12, 14, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(14)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.1)  # Add dropout after pooling
        
        self.conv4 = nn.Conv2d(14, 16, 3)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 18, 3)
        self.bn5 = nn.BatchNorm2d(18)
        
        self.conv6 = nn.Conv2d(18, 10, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)  # Regularization
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.dropout2(x)  # Regularization
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

class Model_3(nn.Module):
    """
    Model_3: Optimize Training Dynamics
    
    Target:
    Add LR scheduler for better convergence and stability. Fine-tune dropout to 
    balance regularization and capacity. Achieve consistent 99.4%+ accuracy in 
    last few epochs.
    
    Results:
    Parameters: 7,450 (same architecture)
    Best Training Accuracy: 99.18%
    Best Test Accuracy: 99.38%
    Epochs to converge: 15
    Consistency: Last 5 epochs: [99.26, 99.25, 99.30, 99.38, 99.26] - MISSED 99.4%
    
    Analysis:
    The LR scheduler (StepLR) and reduced dropout (0.05) improved stability and 
    pushed accuracy higher. We're getting very close to 99.4% (best: 99.38%) but 
    missing the consistency requirement. The model shows stable training with good 
    convergence. We need a final push to cross 99.4% barrier consistently.
    """
    def __init__(self):
        super(Model_3, self).__init__()
        # Same architecture as Model_2 with tuned dropout
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.05)  # Reduced dropout for consistency
        
        self.conv3 = nn.Conv2d(12, 14, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(14)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.05)  # Reduced dropout for consistency
        
        self.conv4 = nn.Conv2d(14, 16, 3)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 18, 3)
        self.bn5 = nn.BatchNorm2d(18)
        
        self.conv6 = nn.Conv2d(18, 10, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)
