#!/usr/bin/env python3
"""
MNIST Efficient CNN Training Script
ERA Session 4 Assignment

This script trains an efficient CNN on MNIST dataset with the following constraints:
- < 25,000 parameters
- ≥ 95% test accuracy in 1 epoch

Author: ERA Student
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class EfficientMNIST(nn.Module):
    """
    Efficient CNN for MNIST classification
    
    Architecture follows ERA Session 4 principles:
    - Progressive feature extraction: Edges → Textures → Patterns → Parts → Objects
    - Hardware-optimized 3x3 kernels
    - Parameter-efficient design with Global Average Pooling
    """
    
    def __init__(self):
        super(EfficientMNIST, self).__init__()
        
        # Feature Extraction Block 1: Edge and Gradient Detection
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Transition Block 1: Reduce spatial dimensions
        self.conv3 = nn.Conv2d(16, 8, 1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Feature Extraction Block 2: Texture and Pattern Detection
        self.conv4 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 24, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(24)
        
        # Transition Block 2
        self.conv6 = nn.Conv2d(24, 12, 1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Feature Extraction Block 3: Part Detection
        self.conv7 = nn.Conv2d(12, 16, 3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.Conv2d(16, 20, 3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(20)
        
        # Global Context Block: Object Detection
        self.conv9 = nn.Conv2d(20, 16, 3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(16)
        
        # Final prediction layer
        self.conv10 = nn.Conv2d(16, 10, 1, bias=False)
        
        # Global Average Pooling instead of FC layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Block 1: Edge and Gradient Detection
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        
        # Transition 1
        x = self.pool1(self.conv3(x))
        
        # Block 2: Texture and Pattern Detection
        x = self.dropout(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(F.relu(self.bn5(self.conv5(x))))
        
        # Transition 2
        x = self.pool2(self.conv6(x))
        
        # Block 3: Part Detection
        x = self.dropout(F.relu(self.bn7(self.conv7(x))))
        x = self.dropout(F.relu(self.bn8(self.conv8(x))))
        
        # Global Context
        x = self.dropout(F.relu(self.bn9(self.conv9(x))))
        
        # Final prediction
        x = self.conv10(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    """Count total trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_data_loaders(batch_size=128):
    """Prepare MNIST data loaders with augmentation"""
    
    # Training transforms with moderate augmentation
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-3.0, 3.0), fill=(0,)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test transforms (no augmentation)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and create datasets
    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    return train_loader, test_loader

def get_correct_predictions(prediction, labels):
    """Calculate number of correct predictions"""
    return prediction.argmax(dim=1).eq(labels).sum().item()

def train_epoch(model, device, train_loader, optimizer, criterion, scheduler=None):
    """Train model for one epoch"""
    model.train()
    pbar = tqdm(train_loader, desc='Training')
    
    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        correct += get_correct_predictions(output, target)
        processed += len(data)
        
        accuracy = 100 * correct / processed
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_description(
            f'Loss: {loss.item():.4f} | Acc: {accuracy:.2f}% | LR: {current_lr:.6f}'
        )
    
    avg_loss = train_loss / len(train_loader)
    accuracy = 100 * correct / processed
    
    return avg_loss, accuracy

def test_epoch(model, device, test_loader, criterion):
    """Evaluate model on test data"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            correct += get_correct_predictions(output, target)
    
    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)
    
    return avg_loss, accuracy

def evaluate_clean_train_data(model, device, batch_size=128):
    """Evaluate model on clean training data (no augmentation) for fair comparison"""
    # Clean training data (no augmentation)
    clean_train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    clean_train_data = datasets.MNIST('./data', train=True, download=False, transform=clean_train_transforms)
    clean_train_loader = torch.utils.data.DataLoader(
        clean_train_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in clean_train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            correct += get_correct_predictions(output, target)
            total += target.size(0)
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    """Main training function"""
    
    print("="*60)
    print("MNIST Efficient CNN - ERA Session 4 Assignment")
    print("="*60)
    print(f"Target: < 25K parameters, ≥95% accuracy in 1 epoch")
    print(f"Device: {device}")
    print("="*60)
    
    # Create model and check parameters
    model = EfficientMNIST().to(device)
    total_params = count_parameters(model)
    
    print(f"Model: EfficientMNIST")
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter constraint (< 25K): {'✅ PASSED' if total_params < 25000 else '❌ FAILED'}")
    
    # Get data loaders
    batch_size = 128
    train_loader, test_loader = get_data_loaders(batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Training configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # OneCycleLR for aggressive learning rate scheduling
    total_steps = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=total_steps,
        epochs=1,
        pct_start=0.2,
        anneal_strategy='cos'
    )
    
    print(f"Optimizer: Adam (lr=0.001, weight_decay=1e-4)")
    print(f"Scheduler: OneCycleLR (max_lr=0.01)")
    print("="*60)
    
    # Training
    print("\nStarting training...")
    epoch = 1
    
    train_loss, train_acc_augmented = train_epoch(model, device, train_loader, optimizer, criterion, scheduler)
    test_loss, test_acc = test_epoch(model, device, test_loader, criterion)
    
    # Evaluate on clean training data for fair comparison
    print("\nEvaluating on clean training data...")
    train_acc_clean = evaluate_clean_train_data(model, device, batch_size)
    
    # Results
    print(f"\nEPOCH {epoch} RESULTS:")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc (Augmented): {train_acc_augmented:.2f}%")
    print(f"Train Acc (Clean): {train_acc_clean:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Check targets
    target_achieved = test_acc >= 95.0
    param_constraint = total_params < 25000
    
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    print(f"📊 Parameters: {total_params:,} < 25,000 ({'✅ PASSED' if param_constraint else '❌ FAILED'})")
    print(f"🎯 Test Accuracy: {test_acc:.2f}% ≥ 95% ({'✅ PASSED' if target_achieved else '❌ FAILED'})")
    print(f"📈 Train Accuracy (Clean): {train_acc_clean:.2f}%")
    print(f"📉 Train Accuracy (Augmented): {train_acc_augmented:.2f}%")
    print(f"⏱️  Epochs: 1 (✅ PASSED)")
    print("="*60)
    
    if target_achieved and param_constraint:
        print("🎉 ALL REQUIREMENTS MET! Assignment successful! 🎉")
    else:
        print("❌ Some requirements not met. Please review the architecture.")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_acc,
        'train_accuracy_clean': train_acc_clean,
        'train_accuracy_augmented': train_acc_augmented,
        'total_parameters': total_params,
        'epochs_trained': 1
    }, 'efficient_mnist_model.pth')
    
    print(f"\n✅ Model saved as 'efficient_mnist_model.pth'")
    
    # Save results to JSON
    results = {
        'total_parameters': total_params,
        'test_accuracy': test_acc,
        'train_accuracy_clean': train_acc_clean,
        'train_accuracy_augmented': train_acc_augmented,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'epochs': 1,
        'requirements_met': {
            'parameters_under_25k': param_constraint,
            'accuracy_over_95': target_achieved,
            'single_epoch': True
        },
        'assignment_status': 'PASSED' if (target_achieved and param_constraint) else 'FAILED'
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved as 'training_results.json'")
    print("\nAssignment completed!")

if __name__ == "__main__":
    main()
