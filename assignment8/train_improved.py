#!/usr/bin/env python3
"""
Main training script for ResNet-18 on CIFAR-100
Refactored version with proper configuration management and error handling
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from config import Config, get_device
from data_loader import get_cifar100_loaders
from model import create_model
from trainer import Trainer
from utils import (
    setup_logging, 
    validate_config, 
    count_parameters,
    create_experiment_dir,
    save_training_logs_markdown
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-100')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration JSON file')
    parser.add_argument('--experiment-dir', type=str, default=None,
                       help='Directory to save experiment results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def load_config(config_path: str = None) -> Config:
    """Load configuration from file or use defaults"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
        print(f"Loaded configuration from {config_path}")
    else:
        config = Config()
        print("Using default configuration")
    
    return config

def print_experiment_info(config: Config, device: str, model: torch.nn.Module):
    """Print experiment information"""
    param_info = count_parameters(model)
    
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Device: {device}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Target Accuracy: {config.training.target_accuracy}%")
    print(f"Total Parameters: {param_info['total']:,}")
    print(f"Trainable Parameters: {param_info['trainable']:,}")
    print("=" * 60)

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting ResNet-18 CIFAR-100 training")
    
    try:
        # Load configuration
        config = load_config(args.config)
        validate_config(config.to_dict())
        
        # Setup experiment directory
        if args.experiment_dir:
            exp_dir = args.experiment_dir
            Path(exp_dir).mkdir(parents=True, exist_ok=True)
        else:
            exp_dir = "."
        
        # Auto-detect device
        device = get_device(config.system)
        logger.info(f"Using device: {device}")
        
        # Load data
        logger.info("Loading CIFAR-100 dataset...")
        train_loader, test_loader = get_cifar100_loaders(config.data, config.training.batch_size)
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        # Create model
        logger.info(f"Creating {config.model.name} model...")
        model = create_model(config.model)
        
        # Print experiment info
        print_experiment_info(config, device, model)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            # TODO: Implement resume functionality
            pass
        
        # Start training
        logger.info("Starting training...")
        training_logs = trainer.train(
            epochs=config.training.epochs,
            target_accuracy=config.training.target_accuracy,
            save_model=config.system.save_model
        )
        
        # Save training logs
        log_path = Path(exp_dir) / config.system.log_file_path
        save_training_logs_markdown(training_logs, config.to_dict(), str(log_path))
        
        # Save final config
        config_path = Path(exp_dir) / "final_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        logger.info("Training completed successfully!")
        
        # Print final results
        if training_logs:
            best_epoch = max(training_logs, key=lambda x: x['test_acc'])
            print(f"\nFINAL RESULTS:")
            print(f"Best Test Accuracy: {best_epoch['test_acc']:.2f}% (Epoch {best_epoch['epoch']})")
            print(f"Total Training Time: {sum(log['epoch_time'] for log in training_logs)/60:.1f} minutes")
            
            if best_epoch['test_acc'] >= config.training.target_accuracy:
                print(f"🎯 TARGET ACHIEVED! ({config.training.target_accuracy}% target reached)")
            else:
                print(f"❌ Target not reached ({config.training.target_accuracy}% target)")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
