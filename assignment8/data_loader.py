import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple
import logging

from config import DataConfig

logger = logging.getLogger(__name__)

def get_transforms(config: DataConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create training and testing transforms based on configuration"""
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=config.random_crop_padding),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(config.rotation_degrees),
        transforms.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast,
            saturation=config.color_jitter_saturation,
            hue=config.color_jitter_hue
        ),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])

    # Simple transforms for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])
    
    return transform_train, transform_test

def get_cifar100_loaders(config: DataConfig, batch_size: int = None) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 data loaders with appropriate transforms based on configuration
    
    Args:
        config: DataConfig object containing data loading parameters
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    logger.info(f"Loading {config.dataset_name} dataset from {config.data_dir}")
    
    transform_train, transform_test = get_transforms(config)

    try:
        # Load datasets
        trainset = torchvision.datasets.CIFAR100(
            root=config.data_dir, 
            train=True, 
            download=True, 
            transform=transform_train
        )
        # Use provided batch_size or fall back to a default since DataConfig doesn't have batch_size
        effective_batch_size = batch_size if batch_size is not None else 128
        
        trainloader = DataLoader(
            trainset, 
            batch_size=effective_batch_size, 
            shuffle=True, 
            num_workers=config.num_workers, 
            pin_memory=config.pin_memory
        )

        testset = torchvision.datasets.CIFAR100(
            root=config.data_dir, 
            train=False, 
            download=True, 
            transform=transform_test
        )
        testloader = DataLoader(
            testset, 
            batch_size=effective_batch_size, 
            shuffle=False, 
            num_workers=config.num_workers, 
            pin_memory=config.pin_memory
        )
        
        logger.info(f"Dataset loaded successfully. Train: {len(trainset)}, Test: {len(testset)}")
        return trainloader, testloader
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

# Backward compatibility function
def get_cifar100_loaders_legacy(batch_size=128, num_workers=2):
    """Legacy function for backward compatibility"""
    config = DataConfig()
    # Update the fields that need to be modified from defaults
    # Note: DataConfig uses batch_size from TrainingConfig, so we need to handle this differently
    train_transform, test_transform = get_transforms(config)
    
    # Load datasets directly for legacy support
    import torchvision
    trainset = torchvision.datasets.CIFAR100(
        root=config.data_dir, train=True, download=True, transform=train_transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=config.pin_memory
    )

    testset = torchvision.datasets.CIFAR100(
        root=config.data_dir, train=False, download=True, transform=test_transform
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=config.pin_memory
    )
    
    return trainloader, testloader
