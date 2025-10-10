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
    
    # Check if checkpoint exists
    checkpoint_path = 'best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found!")
        print("Please run train.py first to create the initial checkpoint.")
        return
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with best accuracy {checkpoint['best_acc']:.2f}%")
    
    # Hyperparameters (same as original training)
    batch_size = 128
    learning_rate = 0.1
    weight_decay = 5e-4
    target_epochs = 90  # Continue until epoch 90
    
    # Load data
    print("Loading CIFAR-100 dataset...")
    train_loader, test_loader = get_cifar100_loaders(batch_size=batch_size, num_workers=2)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("-" * 50)
    
    # Create model and load state
    print("Creating ResNet-18 model...")
    model = ResNet18()
    model.load_state_dict(checkpoint['model_state_dict'])
    
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
    
    # Load optimizer state
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available (older checkpoints might not have it)
    if 'scheduler_state_dict' in checkpoint:
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print("No scheduler state in checkpoint, using fresh scheduler from epoch 81")
    
    # Calculate remaining epochs
    start_epoch = checkpoint['epoch']
    remaining_epochs = target_epochs - start_epoch
    
    if remaining_epochs <= 0:
        print(f"Model has already been trained to epoch {start_epoch}.")
        print(f"Target is epoch {target_epochs}. No additional training needed.")
        return
    
    print(f"Continuing training from epoch {start_epoch + 1} to epoch {target_epochs}")
    print(f"Remaining epochs: {remaining_epochs}")
    print("-" * 50)
    
    # Continue training (remove early stopping condition)
    training_logs = trainer.continue_train(
        epochs=remaining_epochs, 
        start_epoch=start_epoch,
        target_accuracy=None  # Remove early stopping
    )
    
    # Append new logs to existing training logs
    append_logs_to_markdown(training_logs, start_epoch)
    
    print(f"\nContinued training completed!")
    print(f"Training continued from epoch {start_epoch + 1} to epoch {target_epochs}")
    print("Check 'training_logs.md' for updated logs.")

def append_logs_to_markdown(new_logs, start_epoch):
    """Append new training logs to existing markdown file"""
    if not new_logs:
        return
    
    # Read existing content
    try:
        with open('training_logs.md', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("No existing training_logs.md found. Creating new one.")
        content = ""
    
    # Find where to insert new logs (before the ## Result section)
    result_section = "## Result"
    if result_section in content:
        # Split content at Result section
        before_result = content[:content.find(result_section)]
        result_and_after = content[content.find(result_section):]
        
        # Add new epoch logs
        new_log_lines = ""
        for log in new_logs:
            new_log_lines += (f"|  {log['epoch']:2d} |   {log['train_loss']:.4f} |       {log['train_acc']:.2f} |  {log['test_loss']:.4f} |      {log['test_acc']:.2f} |    {log['lr']:.6f} |   {log['epoch_time']:.1f} |\n")
        
        # Update the result section with new best accuracy
        all_logs_content = before_result + new_log_lines + "\n" + result_and_after
        
        # Update the result section with final metrics
        best_log = max(new_logs, key=lambda x: x['test_acc'])
        final_log = new_logs[-1]
        
        # Replace result section with updated info
        updated_result = f"""## Result
- **🎯 CONTINUED TRAINING:** Final accuracy {final_log['test_acc']:.2f}% at epoch {final_log['epoch']}
- **Training accuracy:** {final_log['train_acc']:.2f}%
- **Total epochs:** {final_log['epoch']}/90 (continued to target)
- **Best test accuracy:** {best_log['test_acc']:.2f}% (epoch {best_log['epoch']})"""
        
        all_logs_content = all_logs_content.replace(
            result_and_after,
            updated_result + "\n\n" + result_and_after.split("\n\n", 1)[1] if "\n\n" in result_and_after else updated_result
        )
        
        # Write updated content
        with open('training_logs.md', 'w') as f:
            f.write(all_logs_content)
    else:
        print("Could not find Result section in training_logs.md")

if __name__ == "__main__":
    main()
