"""
MNIST CNN Training Module

High-level Functionality:
- Data Loading: MNIST dataset with optional augmentation (rotation)
- Model Training: Unified training function supporting 3 iterative models
- Training Optimization: StepLR scheduler for improved convergence
- Comprehensive Logging: Epoch-wise accuracy tracking and validation metrics

Core Functions:
1. get_data_loaders(): Creates train/test loaders with optional augmentation
2. train_model(): Universal training function with scheduler support
3. main(): Sequential training of Model_1, Model_2, Model_3 with proper evaluation

Target: 99.4%+ validation accuracy consistently achieved with ≤8K parameters in ≤15 epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Model_1, Model_2, Model_3, count_parameters
import numpy as np

transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_data_loaders(batch_size=128, augment=False):
    if augment:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=6),  # Light rotation as per notes.txt
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transform_base
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform_base)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    print(f"Train/Validation split: {len(train_dataset)} / {len(test_dataset)}")
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=15, lr=0.01, weight_decay=1e-4, 
                use_scheduler=False, model_name="Model"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    criterion = nn.NLLLoss()
    best_val_acc = 0
    best_train_acc = 0
    acc_history = []
    train_acc_history = []
    
    for epoch in range(1, epochs+1):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
                
            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)  # Works for both logits and log_softmax
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
        train_acc = 100. * correct / total
        train_acc_history.append(train_acc)
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)  # Works for both logits and log_softmax
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        val_acc = 100. * correct / total
        acc_history.append(val_acc)
        print(f"Epoch {epoch}: Train {train_acc:.2f}% | Val {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        # Check for early stopping if last 5 epochs are consistently ≥99.4%
        if epoch >= 10 and len(acc_history) >= 5:
            last_5 = acc_history[-5:]
            if all([a >= 99.40 for a in last_5]):
                print(f"🎯 Early stopping at epoch {epoch} - Consistent 99.4%+ achieved!")
                break
        

                
        if use_scheduler:
            scheduler.step()
    
    print(f"Total epochs: {epoch}")
    print(f"Consistent epochs (last 5): {[round(a,2) for a in acc_history[-5:]]}")
    print(f"Total parameter count: {count_parameters(model)}")
    print(f"Best validation accuracy: {round(best_val_acc,2)}%")
    print(f"Best train accuracy: {round(best_train_acc,2)}%")
    
    # Check consistency requirement
    if len(acc_history) >= 5:
        last_5 = acc_history[-5:]
        consistent = all([a >= 99.40 for a in last_5])
        print(f"Consistency check (last 5 epochs ≥99.4%): {'✅ PASSED' if consistent else '❌ FAILED'}")
    
    return best_val_acc, acc_history, best_train_acc, train_acc_history, epoch

def main():
    print("="*60)
    print("Model 1 Target: 99.4% accuracy, <8K params, ≤15 epochs")
    print("="*60)
    train_loader, test_loader = get_data_loaders(augment=False)
    model1 = Model_1()
    print(f"Model 1 parameter count: {count_parameters(model1)}")
    train_model(model1, train_loader, test_loader, model_name="Model_1")

    print("\n" + "="*60)
    print("Model 2 Target: 99.4%+ accuracy, <8K params, ≤15 epochs")
    print("="*60)
    train_loader, test_loader = get_data_loaders(augment=True)
    model2 = Model_2()
    print(f"Model 2 parameter count: {count_parameters(model2)}")
    train_model(model2, train_loader, test_loader, model_name="Model_2")

    print("\n" + "="*60)
    print("Model 3 Target: Consistent 99.4%+ accuracy, <8K params, ≤15 epochs")
    print("="*60)
    train_loader, test_loader = get_data_loaders(augment=True)
    model3 = Model_3()
    print(f"Model 3 parameter count: {count_parameters(model3)}")
    train_model(model3, train_loader, test_loader, lr=0.01, use_scheduler=True, model_name="Model_3")

if __name__ == "__main__":
    main()
