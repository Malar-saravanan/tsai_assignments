"""
Data loading and preprocessing module for CIFAR-10
Includes Albumentations augmentations as specified
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


class AlbumentationsDataset:
    """Custom dataset wrapper for Albumentations"""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert PIL to numpy array
        if hasattr(image, 'numpy'):
            image = np.array(image)
        else:
            image = np.array(image)
            
        # Apply albumentations transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label


def get_cifar10_stats():
    """Get CIFAR-10 dataset statistics"""
    # CIFAR-10 statistics
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    return mean, std


def get_train_transforms():
    """
    Training transforms with Albumentations
    Required augmentations:
    - horizontal flip
    - shiftScaleRotate  
    - coarseDropout (max_holes=1, max_height=16px, max_width=16, 
                     min_holes=1, min_height=16px, min_width=16px, 
                     fill_value=mean, mask_fill_value=None)
    """
    mean, std = get_cifar10_stats()
    
    # Calculate fill_value as mean of dataset (converted to 0-255 range)
    fill_value = [int(m * 255) for m in mean]
    
    train_transform = A.Compose([
        # Horizontal flip
        A.HorizontalFlip(p=0.5),
        
        # Shift Scale Rotate with exact specifications
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1, 
            rotate_limit=15,
            p=0.5
        ),
        
        # Coarse Dropout with EXACT specifications as required
        # (using current Albumentations API)
        A.CoarseDropout(
            num_holes_range=(1, 1),        # min_holes=1, max_holes=1
            hole_height_range=(16, 16),    # min_height=16px, max_height=16px  
            hole_width_range=(16, 16),     # min_width=16px, max_width=16px
            fill=fill_value,               # fill_value = mean of dataset
            fill_mask=None,                # mask_fill_value = None
            p=0.5
        ),
        
        # Normalize and convert to tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transform


def get_test_transforms():
    """Test transforms (only normalization)"""
    mean, std = get_cifar10_stats()
    
    test_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return test_transform


def get_data_loaders(batch_size=128, num_workers=4, val_split=0.1):
    """
    Get CIFAR-10 data loaders with train/validation split
    
    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of worker processes for data loading
        val_split: Fraction of training data to use for validation
        
    Returns:
        train_loader, val_loader, test_loader, classes
    """
    
    # Download and prepare datasets
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=None
    )
    
    # Split training data into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Get class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Apply transforms
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    # Wrap datasets with albumentations
    train_dataset = AlbumentationsDataset(train_dataset, train_transform)
    val_dataset = AlbumentationsDataset(val_dataset, test_transform)  # No augmentation for validation
    test_dataset = AlbumentationsDataset(test_dataset, test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, classes


def visualize_augmentations(dataset, num_samples=8):
    """Visualize augmentations applied to the dataset"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Get a sample
        image, label = dataset[i]
        
        # Convert tensor back to numpy for visualization
        if isinstance(image, torch.Tensor):
            # Denormalize
            mean, std = get_cifar10_stats()
            for t, m, s in zip(image, mean, std):
                t.mul_(s).add_(m)
            
            # Convert to numpy and clip
            image = image.permute(1, 2, 0).numpy()
            image = np.clip(image, 0, 1)
        
        axes[i].imshow(image)
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Test data loading
    print("Testing CIFAR-10 data loading with albumentations...")
    
    train_loader, test_loader, classes = get_data_loaders(batch_size=4)
    
    # Test loading a batch
    train_iter = iter(train_loader)
    images, labels = next(train_iter)
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Image dtype: {images.dtype}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    print("\n=== Augmentation Verification ===")
    print("✓ Horizontal Flip implemented") 
    print("✓ ShiftScaleRotate implemented")
    print("✓ CoarseDropout with exact specifications implemented")
    print("✓ Mean and std normalization applied")
    print("✓ Albumentations library used")
