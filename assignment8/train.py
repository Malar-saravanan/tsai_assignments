import torch
import os
from datetime import datetime
from model import ResNet18
from data_loader import get_cifar100_loaders
from trainer import Trainer

def main():
    # Check for MPS availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    print(f"PyTorch version: {torch.__version__}")
    print("-" * 50)
    
    # Hyperparameters optimized for quick training
    batch_size = 128
    learning_rate = 0.1
    weight_decay = 5e-4
    epochs = 100
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    train_loader, test_loader = get_cifar100_loaders(batch_size=batch_size, num_workers=2)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: 100")
    print("-" * 50)
    
    # Create model
    print("Creating ResNet-18 model...")
    model = ResNet18()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("-" * 50)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Start training
    training_logs = trainer.train(epochs=epochs)
    
    # Save training logs to markdown
    save_logs_to_markdown(training_logs)
    
    print("\nTraining completed!")
    print("Check 'training_logs.md' for detailed logs.")

def save_logs_to_markdown(logs):
    """Save training logs to a markdown file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open('training_logs.md', 'w') as f:
        f.write("# ResNet-18 CIFAR-100 Training Logs\n\n")
        f.write(f"**Training started:** {timestamp}\n\n")
        f.write(f"**Model:** ResNet-18\n")
        f.write(f"**Dataset:** CIFAR-100\n")
        f.write(f"**Target Accuracy:** 73%\n\n")
        
        f.write("## Training Configuration\n\n")
        f.write("- **Architecture:** ResNet-18 (Custom implementation)\n")
        f.write("- **Optimizer:** SGD with momentum=0.9\n")
        f.write("- **Learning Rate:** 0.1 (initial)\n")
        f.write("- **Scheduler:** CosineAnnealingLR\n")
        f.write("- **Weight Decay:** 5e-4\n")
        f.write("- **Batch Size:** 128\n")
        f.write("- **Data Augmentation:** RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter\n\n")
        
        f.write("## Epoch-by-Epoch Results\n\n")
        f.write("| Epoch | Train Loss | Train Acc (%) | Test Loss | Test Acc (%) | Learning Rate | Time (s) |\n")
        f.write("|-------|------------|---------------|-----------|--------------|---------------|----------|\n")
        
        for log in logs:
            f.write(f"| {log['epoch']:3d} | "
                   f"{log['train_loss']:8.4f} | "
                   f"{log['train_acc']:11.2f} | "
                   f"{log['test_loss']:7.4f} | "
                   f"{log['test_acc']:10.2f} | "
                   f"{log['lr']:11.6f} | "
                   f"{log['epoch_time']:6.1f} |\n")
        
        # Summary
        if logs:
            best_epoch = max(logs, key=lambda x: x['test_acc'])
            f.write(f"\n## Summary\n\n")
            f.write(f"- **Best Test Accuracy:** {best_epoch['test_acc']:.2f}% (Epoch {best_epoch['epoch']})\n")
            f.write(f"- **Final Test Accuracy:** {logs[-1]['test_acc']:.2f}%\n")
            f.write(f"- **Total Epochs:** {len(logs)}\n")
            f.write(f"- **Total Training Time:** {sum(log['epoch_time'] for log in logs)/60:.1f} minutes\n")
            
            if best_epoch['test_acc'] >= 73.0:
                f.write(f"- **Target Achieved:** ✅ Yes (73% target reached)\n")
            else:
                f.write(f"- **Target Achieved:** ❌ No (73% target not reached)\n")

if __name__ == "__main__":
    main()
