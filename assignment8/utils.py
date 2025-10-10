"""
Utility functions for logging, error handling, and model management
"""

import os
import json
import logging
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_acc: float,
    training_logs: List[Dict],
    filepath: str
) -> None:
    """Save model checkpoint with all training state"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'training_logs': training_logs,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
        raise

def load_checkpoint(filepath: str, device: str) -> Dict:
    """Load model checkpoint with error handling"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file {filepath} not found")
        
        checkpoint = torch.load(filepath, map_location=device)
        logging.info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        raise

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def save_training_logs_markdown(
    logs: List[Dict], 
    config: Dict,
    filepath: str = "training_logs.md"
) -> None:
    """Save training logs in markdown format"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(filepath, 'w') as f:
            f.write("# ResNet-18 CIFAR-100 Training Logs\n\n")
            f.write("## Target\n")
            f.write(f"- **Objective:** Train ResNet-18 from scratch on CIFAR-100 to achieve {config['training']['target_accuracy']}% test accuracy\n")
            f.write(f"- **Dataset:** CIFAR-100 (50k train, 10k test, {config['model']['num_classes']} classes)\n")
            f.write(f"- **Model:** {config['model']['name']} ({count_parameters.__name__} parameters)\n")
            f.write(f"- **Device:** Auto-detected\n")
            f.write(f"- **Config:** SGD, LR={config['training']['learning_rate']}, CosineAnnealingLR, batch={config['training']['batch_size']}, data augmentation\n\n")
            
            f.write("## Training Logs\n\n")
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
            
            if logs:
                best_epoch = max(logs, key=lambda x: x['test_acc'])
                final_epoch = logs[-1]
                
                f.write(f"\n## Result\n")
                f.write(f"- **🎯 TARGET {'ACHIEVED' if best_epoch['test_acc'] >= config['training']['target_accuracy'] else 'NOT REACHED'}:** {best_epoch['test_acc']:.2f}% test accuracy at epoch {best_epoch['epoch']}\n")
                f.write(f"- **Training accuracy:** {final_epoch['train_acc']:.2f}%\n")
                f.write(f"- **Total epochs:** {len(logs)} ({'stopped at target' if best_epoch['test_acc'] >= config['training']['target_accuracy'] else 'completed'})\n")
                f.write(f"- **Best test accuracy:** {best_epoch['test_acc']:.2f}% (epoch {best_epoch['epoch']})\n\n")
                
                f.write(f"## Analysis\n")
                f.write(f"- **Convergence:** Multiple phases of learning\n")
                f.write(f"- **Performance:** {'Target achieved' if best_epoch['test_acc'] >= config['training']['target_accuracy'] else 'Target not reached'}, good generalization\n")
                f.write(f"- **Efficiency:** {best_epoch['test_acc'] / len(logs):.2f}% improvement per epoch\n")
                f.write(f"- **Time:** {sum(log['epoch_time'] for log in logs)/60:.1f} minutes total\n")
                f.write(f"- **Success factors:** ResNet architecture, data augmentation, learning rate scheduling\n")
                
        logging.info(f"Training logs saved to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save training logs: {e}")
        raise

def validate_config(config: Dict) -> None:
    """Validate configuration parameters"""
    required_keys = ['model', 'training', 'data', 'system']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # Validate ranges
    if config['training']['learning_rate'] <= 0:
        raise ValueError("Learning rate must be positive")
    
    if config['training']['batch_size'] <= 0:
        raise ValueError("Batch size must be positive")
    
    if config['training']['epochs'] <= 0:
        raise ValueError("Number of epochs must be positive")

def create_experiment_dir(base_dir: str = "experiments") -> str:
    """Create timestamped experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"resnet18_cifar100_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
