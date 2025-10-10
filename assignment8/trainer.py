import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import time
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from config import Config, TrainingConfig
from utils import save_checkpoint, save_training_logs_markdown

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model: nn.Module, device: str, train_loader, test_loader, 
                 config: Config):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=config.training.learning_rate,
            momentum=config.training.momentum, 
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.training.epochs, 
            eta_min=1e-6
        )
        
        # Training logs
        self.training_logs = []
        
        logger.info(f"Trainer initialized with device: {device}")
        logger.info(f"Optimizer: SGD (lr={config.training.learning_rate}, momentum={config.training.momentum}, wd={config.training.weight_decay})")
        logger.info(f"Scheduler: CosineAnnealingLR (T_max={config.training.epochs})")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_time = time.time() - start_time
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc, epoch_time
    
    def test(self):
        """Evaluate on test set"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
    
    def train(self, epochs: int = None, target_accuracy: float = None, save_model: bool = True):
        """Full training loop"""
        if epochs is None:
            epochs = self.config.training.epochs
        if target_accuracy is None:
            target_accuracy = self.config.training.target_accuracy
            
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training for {epochs} epochs (target: {target_accuracy}%)")
        print(f"Starting training on {self.device}")
        print(f"Training for {epochs} epochs")
        print("-" * 60)
        
        best_acc = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, epoch_time = self.train_epoch(epoch)
            
            # Test
            test_loss, test_acc = self.test()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log results
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'lr': current_lr,
                'epoch_time': epoch_time
            }
            self.training_logs.append(log_entry)
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}% | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
            # Save best model
            if test_acc > best_acc and save_model:
                best_acc = test_acc
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    best_acc=best_acc,
                    training_logs=self.training_logs,
                    filepath=self.config.system.model_save_path
                )
            
            # Early stopping if we reach target accuracy
            if test_acc >= target_accuracy:
                logger.info(f"Reached target accuracy of {target_accuracy}%! Best accuracy: {best_acc:.2f}%")
                print(f"\nReached target accuracy of {target_accuracy}%! Best accuracy: {best_acc:.2f}%")
                break
                
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best test accuracy: {best_acc:.2f}%")
        
        return self.training_logs
    
    def continue_train(self, epochs, start_epoch=0, target_accuracy=None, save_model=True):
        """Continue training from a checkpoint"""
        print(f"Continuing training on {self.device}")
        print(f"Training for {epochs} more epochs (from epoch {start_epoch + 1})")
        print("-" * 60)
        
        best_acc = 0
        # Find current best accuracy from existing logs
        if hasattr(self, 'training_logs') and self.training_logs:
            best_acc = max(log['test_acc'] for log in self.training_logs)
        
        start_time = time.time()
        continue_logs = []
        
        for epoch in range(epochs):
            actual_epoch = start_epoch + epoch + 1
            
            # Train
            train_loss, train_acc, epoch_time = self.train_epoch(actual_epoch - 1)
            
            # Test
            test_loss, test_acc = self.test()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log results
            log_entry = {
                'epoch': actual_epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'lr': current_lr,
                'epoch_time': epoch_time
            }
            continue_logs.append(log_entry)
            
            # Print progress
            print(f"Epoch {actual_epoch:3d}/90 | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}% | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
            # Save best model
            if test_acc > best_acc and save_model:
                best_acc = test_acc
                torch.save({
                    'epoch': actual_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'test_acc': test_acc,
                    'best_acc': best_acc,
                }, 'best_model.pth')
            
            # Optional early stopping
            if target_accuracy and test_acc >= target_accuracy:
                print(f"\nReached target accuracy of {target_accuracy}%! Best accuracy: {best_acc:.2f}%")
                break
                
        total_time = time.time() - start_time
        print(f"\nContinued training completed in {total_time/60:.1f} minutes")
        print(f"Best test accuracy: {best_acc:.2f}%")
        
        return continue_logs
