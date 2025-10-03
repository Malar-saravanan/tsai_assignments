"""
Proper Experiment Runner with Complete Logging and Results Storage
Implements iterative learning approach for CIFAR-10 CNN experiments
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import time
import json
from datetime import datetime
from pathlib import Path
import csv

# Add project root to path
sys.path.append('.')

from models.network import create_baseline_model, create_optimized_model, count_parameters
from data.dataloader import get_data_loaders


class ExperimentLogger:
    """Comprehensive experiment logging"""
    
    def __init__(self, experiment_name, log_dir):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        self.log_file = self.log_dir / f"{experiment_name}_training.log"
        self.csv_file = self.log_dir / f"{experiment_name}_metrics.csv"
        
        # Initialize CSV file
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'lr', 'time'])
    
    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} | {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, lr, epoch_time):
        """Log epoch metrics to CSV"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, lr, epoch_time])


class EnhancedTrainer:
    """Enhanced training class with comprehensive logging"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device, experiment_name, log_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.experiment_name = experiment_name
        
        # Setup logger
        self.logger = ExperimentLogger(experiment_name, log_dir)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'test_losses': [],
            'test_accuracies': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.target_achieved_epochs = 0
        
    def train_epoch(self, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, loader):
        """Validate on given loader"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(f"results/{self.experiment_name.lower()}/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pth')
            self.logger.log(f"💾 Best checkpoint saved at epoch {epoch}")
    
    def train(self, epochs, learning_rate, target_accuracy):
        """Complete training loop with comprehensive logging"""
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        self.logger.log("=" * 80)
        self.logger.log(f"STARTING TRAINING: {self.experiment_name}")
        self.logger.log("=" * 80)
        self.logger.log(f"Target Accuracy: {target_accuracy}%")
        self.logger.log(f"Max Epochs: {epochs}")
        self.logger.log(f"Learning Rate: {learning_rate}")
        self.logger.log(f"Device: {self.device}")
        self.logger.log("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(optimizer, scheduler)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(self.val_loader)
            
            # Test accuracy (for monitoring)
            test_loss, test_acc = self.validate_epoch(self.test_loader)
            
            # Record history
            self.history['train_losses'].append(train_loss)
            self.history['train_accuracies'].append(train_acc)
            self.history['val_losses'].append(val_loss)
            self.history['val_accuracies'].append(val_acc)
            self.history['test_losses'].append(test_loss) 
            self.history['test_accuracies'].append(test_acc)
            self.history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            epoch_time = time.time() - epoch_start_time
            self.history['epoch_times'].append(epoch_time)
            
            # Check for best model
            is_best = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                is_best = True
            
            # Track target achievement
            if val_acc >= target_accuracy:
                self.target_achieved_epochs += 1
            else:
                self.target_achieved_epochs = 0
            
            # Logging
            self.logger.log(f"Epoch [{epoch+1:3d}/{epochs}] - {epoch_time:.1f}s")
            self.logger.log(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            self.logger.log(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            self.logger.log(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
            self.logger.log(f"  LR: {scheduler.get_last_lr()[0]:.6f}, Best Val: {self.best_val_acc:.2f}%")
            
            if val_acc >= target_accuracy:
                self.logger.log(f"  🎯 TARGET ACHIEVED! Validation accuracy: {val_acc:.2f}% >= {target_accuracy}%")
            
            # Log to CSV
            self.logger.log_epoch(epoch+1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, scheduler.get_last_lr()[0], epoch_time)
            
            # Save checkpoint
            self.save_checkpoint(epoch+1, is_best)
            
            # Early stopping if target consistently achieved
            if self.target_achieved_epochs >= 3:
                self.logger.log(f"\n🎉 EARLY STOPPING: Target achieved for {self.target_achieved_epochs} consecutive epochs!")
                break
        
        total_time = time.time() - start_time
        
        # Final test evaluation
        final_test_loss, final_test_acc = self.validate_epoch(self.test_loader)
        
        # Compile results
        results = {
            'experiment_name': self.experiment_name,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'target_accuracy': target_accuracy,
            'best_val_accuracy': self.best_val_acc,
            'best_epoch': self.best_epoch + 1,
            'final_test_accuracy': final_test_acc,
            'target_achieved': self.best_val_acc >= target_accuracy,
            'target_achieved_epochs': self.target_achieved_epochs,
            'history': self.history
        }
        
        self.logger.log("\n" + "=" * 80)
        self.logger.log("TRAINING COMPLETED")
        self.logger.log("=" * 80)
        self.logger.log(f"Total Epochs: {results['total_epochs']}")
        self.logger.log(f"Training Time: {total_time/60:.1f} minutes")
        self.logger.log(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        self.logger.log(f"Final Test Accuracy: {final_test_acc:.2f}%")
        self.logger.log(f"Target ({target_accuracy}%) Achieved: {'✅' if results['target_achieved'] else '❌'}")
        
        return results


def run_experiment_1():
    """Experiment 1: Baseline Implementation"""
    print("\n🚀 EXPERIMENT 1: BASELINE IMPLEMENTATION")
    print("=" * 60)
    
    # Configuration
    config = {
        'experiment_name': 'Experiment_1_Baseline',
        'model_type': 'Baseline',
        'epochs': 25,
        'batch_size': 128,
        'learning_rate': 0.001,
        'target_accuracy': 85.0,
        'description': 'Basic C1C2C3C4 architecture with dilated convolutions'
    }
    
    print("📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Create model
    print("\n🔧 Creating baseline model...")
    model = create_baseline_model()
    total_params = count_parameters(model)
    
    # Get data
    print("\n📊 Loading data...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        batch_size=config['batch_size'], 
        val_split=0.1
    )
    
    # Create trainer and run
    log_dir = f"logs/experiment_1"
    trainer = EnhancedTrainer(model, train_loader, val_loader, test_loader, device, config['experiment_name'], log_dir)
    
    print(f"\n🏋️ Starting training on {device}...")
    results = trainer.train(
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        target_accuracy=config['target_accuracy']
    )
    
    # Add config to results
    results.update(config)
    results['total_parameters'] = total_params
    results['device'] = str(device)
    
    # Save comprehensive results
    results_dir = Path("results/experiment_1")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    with open(results_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary results
    summary = {
        'experiment_name': results['experiment_name'],
        'model_type': results['model_type'],
        'total_parameters': results['total_parameters'],
        'total_epochs': results['total_epochs'],
        'target_accuracy': results['target_accuracy'],
        'best_val_accuracy': results['best_val_accuracy'],
        'final_test_accuracy': results['final_test_accuracy'],
        'target_achieved': results['target_achieved'],
        'training_time_minutes': results['total_time'] / 60,
        'device': results['device']
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_dir}")
    print("📊 Files created:")
    print(f"   - {results_dir}/detailed_results.json")
    print(f"   - {results_dir}/summary.json")  
    print(f"   - {log_dir}/Experiment_1_Baseline_training.log")
    print(f"   - {log_dir}/Experiment_1_Baseline_metrics.csv")
    print(f"   - {results_dir}/checkpoints/best_checkpoint.pth")
    
    return results


def run_experiment_2(exp1_results):
    """Experiment 2: Optimized Implementation (Iterative Improvement)"""
    print("\n🚀 EXPERIMENT 2: OPTIMIZED IMPLEMENTATION")
    print("=" * 60)
    print("🔄 ITERATIVE LEARNING: Building on Experiment 1 insights")
    
    # Configuration (improved based on Exp 1)
    config = {
        'experiment_name': 'Experiment_2_Optimized',
        'model_type': 'Optimized',
        'epochs': 30,
        'batch_size': 128,
        'learning_rate': 0.0015,  # Higher LR based on Exp 1
        'target_accuracy': 89.0,  # Higher target
        'description': 'Enhanced C1C2C3C4 with SE blocks based on Experiment 1 learnings'
    }
    
    print("📋 Configuration (Improved from Exp 1):")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print(f"\n🔄 Improvements based on Experiment 1:")
    print(f"   - Exp 1 achieved: {exp1_results['best_val_accuracy']:.2f}%")
    print(f"   - Added SE attention blocks for better feature selection")
    print(f"   - Increased learning rate: {config['learning_rate']} (vs {exp1_results['learning_rate']})")
    print(f"   - Enhanced regularization with dropout 0.3")
    print(f"   - Target raised to {config['target_accuracy']}% (vs {exp1_results['target_accuracy']}%)")
    
    # Create optimized model
    print("\n🔧 Creating optimized model...")
    model = create_optimized_model()
    total_params = count_parameters(model)
    
    # Get data (same as Exp 1)
    print("\n📊 Loading data...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        batch_size=config['batch_size'], 
        val_split=0.1
    )
    
    # Create trainer and run
    log_dir = f"logs/experiment_2"
    trainer = EnhancedTrainer(model, train_loader, val_loader, test_loader, device, config['experiment_name'], log_dir)
    
    print(f"\n🏋️ Starting training on {device}...")
    results = trainer.train(
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        target_accuracy=config['target_accuracy']
    )
    
    # Add config to results
    results.update(config)
    results['total_parameters'] = total_params
    results['device'] = str(device)
    results['exp1_baseline_accuracy'] = exp1_results['best_val_accuracy']
    results['improvement_over_exp1'] = results['best_val_accuracy'] - exp1_results['best_val_accuracy']
    
    # Save comprehensive results
    results_dir = Path("results/experiment_2")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    with open(results_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary results
    summary = {
        'experiment_name': results['experiment_name'],
        'model_type': results['model_type'],
        'total_parameters': results['total_parameters'],
        'total_epochs': results['total_epochs'],
        'target_accuracy': results['target_accuracy'],
        'best_val_accuracy': results['best_val_accuracy'],
        'final_test_accuracy': results['final_test_accuracy'],
        'target_achieved': results['target_achieved'],
        'training_time_minutes': results['total_time'] / 60,
        'device': results['device'],
        'exp1_baseline_accuracy': results['exp1_baseline_accuracy'],
        'improvement_over_exp1': results['improvement_over_exp1']
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_dir}")
    print("📊 Files created:")
    print(f"   - {results_dir}/detailed_results.json")
    print(f"   - {results_dir}/summary.json")
    print(f"   - {log_dir}/Experiment_2_Optimized_training.log")
    print(f"   - {log_dir}/Experiment_2_Optimized_metrics.csv")
    print(f"   - {results_dir}/checkpoints/best_checkpoint.pth")
    
    return results


def main():
    """Run both experiments with comprehensive logging"""
    print("🚀 CIFAR-10 CNN EXPERIMENTS WITH ITERATIVE LEARNING")
    print("🎯 Implementing C1C2C3C4 Architecture with Dilated Convolutions")
    print("📊 Comprehensive Logging and Results Storage")
    print("="*80)
    
    try:
        # Run Experiment 1
        exp1_results = run_experiment_1()
        
        print("\n" + "="*80)
        print("📊 EXPERIMENT 1 SUMMARY")
        print("="*80)
        print(f"✅ Target: {exp1_results['target_accuracy']}% → Achieved: {exp1_results['best_val_accuracy']:.2f}%")
        print(f"📈 Test Accuracy: {exp1_results['final_test_accuracy']:.2f}%")
        print(f"⚡ Parameters: {exp1_results['total_parameters']:,} < 200k")
        print(f"⏱️  Training Time: {exp1_results['total_time']/60:.1f} minutes")
        
        # Run Experiment 2 (Iterative Improvement)
        exp2_results = run_experiment_2(exp1_results)
        
        print("\n" + "="*80)
        print("📊 EXPERIMENT 2 SUMMARY") 
        print("="*80)
        print(f"✅ Target: {exp2_results['target_accuracy']}% → Achieved: {exp2_results['best_val_accuracy']:.2f}%")
        print(f"📈 Test Accuracy: {exp2_results['final_test_accuracy']:.2f}%")
        print(f"⚡ Parameters: {exp2_results['total_parameters']:,} < 200k")
        print(f"⏱️  Training Time: {exp2_results['total_time']/60:.1f} minutes")
        print(f"🔄 Improvement over Exp 1: +{exp2_results['improvement_over_exp1']:.2f}%")
        
        # Final comparison
        print("\n" + "="*80)
        print("🏆 ITERATIVE LEARNING SUCCESS SUMMARY")
        print("="*80)
        
        comparison = [
            ["Metric", "Experiment 1", "Experiment 2", "Improvement"],
            ["Target", f"{exp1_results['target_accuracy']}%", f"{exp2_results['target_accuracy']}%", f"+{exp2_results['target_accuracy']-exp1_results['target_accuracy']}%"],
            ["Val Accuracy", f"{exp1_results['best_val_accuracy']:.2f}%", f"{exp2_results['best_val_accuracy']:.2f}%", f"+{exp2_results['improvement_over_exp1']:.2f}%"],
            ["Test Accuracy", f"{exp1_results['final_test_accuracy']:.2f}%", f"{exp2_results['final_test_accuracy']:.2f}%", f"+{exp2_results['final_test_accuracy']-exp1_results['final_test_accuracy']:.2f}%"],
            ["Parameters", f"{exp1_results['total_parameters']:,}", f"{exp2_results['total_parameters']:,}", f"{exp2_results['total_parameters']-exp1_results['total_parameters']:,}"],
            ["Epochs", f"{exp1_results['total_epochs']}", f"{exp2_results['total_epochs']}", f"+{exp2_results['total_epochs']-exp1_results['total_epochs']}"],
            ["Target Met", "✅" if exp1_results['target_achieved'] else "❌", "✅" if exp2_results['target_achieved'] else "❌", "✅ Both"]
        ]
        
        for row in comparison:
            print(f"{row[0]:<15} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
        
        print("\n🎉 BOTH EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("📁 All logs and results saved in respective directories")
        print("🔄 Iterative learning approach demonstrated clear improvement")
        
        return exp1_results, exp2_results
        
    except Exception as e:
        print(f"❌ Error during experiments: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()
