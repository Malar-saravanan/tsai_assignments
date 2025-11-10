"""
ResNet50 training for 75%+ accuracy on ImageNet
Adopted best practices from reference: AutoAugment, RandomErasing, better mixed precision
"""
import os
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler  # Updated to torch.amp (PyTorch 2.0+)

from model import create_resnet50
from data import ImageNetDataModule
from utils import (
    AverageMeter, accuracy, save_checkpoint, load_checkpoint,
    LabelSmoothingCrossEntropy, adjust_learning_rate, Mixup,
    coerce_labels_to_long_tensor  # Added from reference for robustness
)


def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, device, 
                mixup=None, random_erasing_transform=None, amp_enabled=True, amp_dtype=torch.float16):
    """
    Training epoch with mixed precision, mixup, and random erasing
    Adopted from reference: RandomErasing adds ~0.5% accuracy
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        
        # Apply RandomErasing per image (from reference - adds ~0.5% accuracy)
        if random_erasing_transform is not None:
            for j in range(images.size(0)):
                images[j] = random_erasing_transform(images[j])
        
        # Coerce labels to LongTensor for robustness (from reference)
        target = coerce_labels_to_long_tensor(target)
        target = target.to(device, non_blocking=True)

        # Apply mixup if enabled
        if mixup is not None and torch.rand(1) < 0.5:
            images, targets_a, targets_b, lam = mixup(images, target)
            
            # Better mixed precision with explicit device_type and dtype (from reference)
            with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                output = model(images)
                loss = lam * criterion(output, targets_a) + (1 - lam) * criterion(output, targets_b)
            
            targets_for_acc = target  # Use original target for accuracy
        else:
            with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                output = model(images)
                loss = criterion(output, target)
            
            targets_for_acc = target

        # Backward pass with conditional gradient scaling (from reference)
        optimizer.zero_grad(set_to_none=True)  # More efficient
        
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets_for_acc, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

    return top1.avg, losses.avg


def validate(model, val_loader, criterion, device, amp_enabled=True, amp_dtype=torch.float16):
    """
    Validation with mixed precision
    Updated from reference with better mixed precision handling
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            
            # Coerce labels for robustness (from reference)
            target = coerce_labels_to_long_tensor(target)
            target = target.to(device, non_blocking=True)

            # Better mixed precision (from reference)
            with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                output = model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print(f'Test: [{i}/{len(val_loader)}] '
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg


def main():
    parser = argparse.ArgumentParser(description='ResNet50 ImageNet Training')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to ImageNet dataset')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true',
                        help='Only run validation')
    parser.add_argument('--epochs', type=int, default=75,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Fine-tuning mode with reduced learning rate')
    # Added from reference: configurable precision for better mixed precision control
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Numerical precision: fp32 (no AMP), fp16 (AMP+GradScaler), bf16 (AMP no scaler)')
    parser.add_argument('--use-official-resnet', action='store_true', default=True,
                        help='Use official torchvision ResNet50 (more reliable, from reference)')
    args = parser.parse_args()

    # BEST-IN-CLASS hyperparameters for 75%+ accuracy under $15 budget
    epochs = args.epochs
    batch_size = 128  # Optimal for Tesla T4 (16GB VRAM)
    learning_rate = args.lr
    momentum = 0.9
    weight_decay = 1e-4  # Critical for generalization
    label_smoothing = 0.1  # Proven to improve top-1 accuracy by 0.5-1%
    mixup_alpha = 0.2  # Data augmentation boost: +0.5-1% accuracy
    
    # Advanced optimization settings for maximum efficiency
    warmup_epochs = 5  # Gradual LR warmup prevents early divergence
    cosine_decay = True  # Better than step decay for final accuracy
    
    # Adjust for fine-tuning mode
    if args.fine_tune:
        print("Fine-tuning mode enabled")
        learning_rate = learning_rate * 0.1  # Reduce LR for fine-tuning
        epochs = min(epochs, 10)  # Limit epochs for fine-tuning
        mixup_alpha = 0.0  # Disable mixup for fine-tuning

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model setup with MAXIMUM optimization for Tesla T4
    # Option to use official torchvision ResNet50 (from reference - more reliable)
    model = create_resnet50(num_classes=1000, use_pretrained_architecture=args.use_official_resnet)
    model = model.to(device, memory_format=torch.channels_last)
    
    # Enable compilation for PyTorch 2.0+ (10-20% speed boost)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled for extra performance")
        except:
            pass
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet50 ({total_params:,} parameters)")
    if args.use_official_resnet:
        print("Using official torchvision ResNet50 architecture (from reference)")

    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                         momentum=momentum, weight_decay=weight_decay)

    # Mixed precision setup (from reference - better handling)
    if args.precision == "fp32":
        amp_enabled = False
        amp_dtype = torch.float16  # Unused but set for compatibility
        print("Using FP32 (no mixed precision)")
    elif args.precision == "bf16":
        amp_enabled = device.type == "cuda"
        amp_dtype = torch.bfloat16
        print("Using BFloat16 mixed precision (no gradient scaling)")
    else:  # fp16
        amp_enabled = device.type == "cuda"
        amp_dtype = torch.float16
        print("Using FP16 mixed precision with gradient scaling")
    
    # Conditional GradScaler - ONLY for fp16! (from reference)
    scaler = GradScaler(enabled=(args.precision == "fp16" and amp_enabled))

    # Data loading with MAXIMUM efficiency for EC2
    # Use all CPU cores for data loading (critical for GPU utilization)
    import os
    num_workers = min(8, os.cpu_count())  # g4dn.xlarge has 4 vCPUs, use 8 workers
    
    # Enhanced data module with AutoAugment and RandomErasing (from reference)
    data_module = ImageNetDataModule(
        args.data_dir, 
        batch_size=batch_size, 
        num_workers=num_workers,
        use_autoaugment=True,  # Adds ~1-2% accuracy
        random_erasing_prob=0.1  # Adds ~0.5% accuracy
    )
    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()
    random_erasing_transform = data_module.random_erasing_transform
    
    print(f"🚀 Data loading: {num_workers} workers, batch size {batch_size}")
    print(f"📦 Training batches per epoch: {len(train_loader):,}")
    print(f"Total training samples: {len(train_loader) * batch_size:,}")
    print(f"✨ AutoAugment: Enabled (+1-2% accuracy)")
    print(f"✨ RandomErasing: {'Enabled' if random_erasing_transform else 'Disabled'} (+0.5% accuracy)")

    # Mixup augmentation
    mixup = Mixup(alpha=mixup_alpha)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            start_epoch, best_acc1 = load_checkpoint(args.resume, model, optimizer)
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # Test only mode
    if args.test_only:
        acc1 = validate(model, val_loader, criterion, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        return

    # Setup comprehensive logging
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create detailed training log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    def log_and_print(message):
        """Log to both console and file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"[{datetime.now()}] {message}\n")
            f.flush()
    
    log_and_print(f"Starting ResNet50 training")
    log_and_print(f"Device: {device}, Batch size: {batch_size}")
    log_and_print(f"Epochs: {start_epoch}-{epochs}" + 
                 (f", Resume from: {best_acc1:.1f}%" if best_acc1 > 0 else ""))
    log_and_print("-" * 50)
    
    # Training start time for budget tracking
    training_start = time.time()
    
    for epoch in range(start_epoch, epochs):
        # Epoch timing and progress tracking
        epoch_start = time.time()
        
        # Adjust learning rate with best-in-class schedule
        current_lr = adjust_learning_rate(optimizer, epoch, learning_rate, 
                                        total_epochs=epochs, warmup_epochs=warmup_epochs)

        # Train for one epoch with enhanced augmentation (from reference)
        train_acc1, train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, device, 
            mixup=mixup,
            random_erasing_transform=random_erasing_transform,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype
        )
        
        epoch_time = time.time() - epoch_start
        hours_elapsed = (time.time() - training_start) / 3600
        
        log_and_print(f'Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f}, '
                     f'Train Acc {train_acc1:.1f}%, Time {epoch_time/60:.1f}min')

        # Enhanced checkpointing strategy
        checkpoint_data = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'train_acc1': train_acc1,
            'train_loss': train_loss,
            'learning_rate': current_lr,
            'timestamp': datetime.now().isoformat(),
            'total_time_hours': hours_elapsed
        }
        
        # Save checkpoint every epoch after 10 (critical for recovery)
        if epoch >= 10:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(checkpoint_data, checkpoint_path)
            
            # Keep only last 5 checkpoints to save disk space
            if epoch >= 15:
                old_checkpoint = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch-5}.pth')
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
        
        # Comprehensive validation and checkpointing
        # IMPORTANT: Validate EVERY epoch (from reference - critical for finding best model)
        log_and_print(f"🔍 Running validation for epoch {epoch+1}...")
        val_start = time.time()
        acc1 = validate(model, val_loader, criterion, device, 
                       amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        val_time = time.time() - val_start
        
        # Update checkpoint data with validation results
        checkpoint_data['val_acc1'] = acc1
        checkpoint_data['val_time_minutes'] = val_time / 60
        
        # Track best model
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        checkpoint_data['best_acc1'] = best_acc1
        
        # Structured validation logging
        log_and_print(f'VALIDATION_RESULTS Epoch {epoch+1}/{epochs}:')
        log_and_print(f'  Val_Acc@1: {acc1:.3f}%')
        log_and_print(f'  Best_Acc@1: {best_acc1:.3f}%') 
        log_and_print(f'  Val_Time: {val_time/60:.1f}min')
        log_and_print(f'  Total_Time: {hours_elapsed:.2f}h')
        
        # Cost estimation (g4dn.xlarge ≈ $0.526/hour, spot ≈ $0.158/hour)
        estimated_cost_ondemand = hours_elapsed * 0.526
        estimated_cost_spot = hours_elapsed * 0.158
        log_and_print(f'  Est_Cost_OnDemand: ${estimated_cost_ondemand:.2f}')
        log_and_print(f'  Est_Cost_Spot: ${estimated_cost_spot:.2f}')
        
        # Save validation checkpoint
        save_checkpoint(checkpoint_data, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))

        # Save best model with comprehensive metadata
        if is_best:
            best_model_data = checkpoint_data.copy()
            best_model_data['best_model_epoch'] = epoch + 1
            best_model_data['is_best'] = True
            save_checkpoint(best_model_data, os.path.join(args.output_dir, 'best_model.pth'))
            log_and_print(f'✅ BEST_MODEL_SAVED: {best_acc1:.3f}% at epoch {epoch+1}')

        # TARGET ACHIEVEMENT CHECKS (preserved from original - your benefit)
        if best_acc1 >= 77.0:
            log_and_print(f"🎉 EXCELLENT! Target 77%+ reached: {best_acc1:.3f}%")
            log_and_print("Stopping training to save budget.")
            break
        elif best_acc1 >= 75.0:
            log_and_print(f"✅ SUCCESS! Target 75%+ reached: {best_acc1:.3f}%")
            log_and_print("Continue training for potential 77%+ (budget permitting)")
        
        # PROGRESS WARNING SYSTEM (preserved from original - your benefit)
        if epoch >= 30 and best_acc1 < 65.0:
            log_and_print(f"⚠️ WARNING: Low accuracy at epoch {epoch+1}: {best_acc1:.3f}%")
            log_and_print("Consider adjusting learning rate or check for issues")
        elif epoch >= 50 and best_acc1 < 70.0:
            log_and_print(f"⚠️ WARNING: May not reach 75%+ target")
            log_and_print(f"Current: {best_acc1:.3f}%, consider fallback strategies")
        
        # EMERGENCY PROTOCOLS (preserved from original - your benefit)
        if epoch >= 60 and best_acc1 >= 72.0 and best_acc1 < 75.0:
            log_and_print(f"🚨 EMERGENCY PROTOCOL: Very close to target!")
            log_and_print(f"Current: {best_acc1:.3f}%, implementing aggressive fine-tuning")
            # Reduce learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            log_and_print(f"Learning rate reduced to: {optimizer.param_groups[0]['lr']:.6f}")
        
        # BUDGET EMERGENCY STOP (preserved from original - your benefit)
        if epoch >= 70 and best_acc1 >= 73.0:
            log_and_print(f"💰 BUDGET ALERT: Saving current best model (73%+)")
            log_and_print(f"This may be sufficient for assignment requirements")
            # Additional validation for confirmation
            log_and_print("Running extended validation to verify accuracy...")
            acc1_extended = validate(model, val_loader, criterion, device,
                                   amp_enabled=amp_enabled, amp_dtype=amp_dtype)
            if acc1_extended >= 74.5:
                log_and_print(f"Extended validation confirms: {acc1_extended:.3f}%")
                log_and_print("This should meet assignment requirements!")
        
        # BUDGET OVERFLOW PROTECTION (preserved from original - your benefit)
        if estimated_cost_ondemand > 16.0:
            log_and_print(f"💸 BUDGET WARNING: Cost ${estimated_cost_ondemand:.2f} approaching $18 limit")
            if best_acc1 >= 73.0:
                log_and_print("Stopping training - good model achieved within budget")
                break
        
        log_and_print("-" * 60)

    # Final training summary
    total_time = time.time() - training_start
    final_cost = total_time / 3600 * 0.526
    
    log_and_print("="*60)
    log_and_print("🏁 TRAINING COMPLETED!")
    log_and_print(f"Final Results:")
    log_and_print(f"   Best Accuracy: {best_acc1:.3f}%")
    log_and_print(f"   Total Time: {total_time/3600:.2f} hours")
    log_and_print(f"   Estimated Cost: ${final_cost:.2f}")
    log_and_print(f"   Target Achievement: {'SUCCESS' if best_acc1 >= 75.0 else 'MISSED'}")
    log_and_print(f"   Budget Status: {'UNDER BUDGET' if final_cost <= 15.0 else 'OVER BUDGET' if final_cost <= 18.0 else 'EXCEEDED'}")
    
    # Save final training summary
    summary_file = os.path.join(args.output_dir, 'training_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"ResNet50 ImageNet Training Summary\n")
        f.write(f"==================================\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(f"Best Accuracy: {best_acc1:.3f}%\n")
        f.write(f"Total Epochs: {epoch + 1}\n")
        f.write(f"Training Time: {total_time/3600:.2f} hours\n")
        f.write(f"Estimated Cost: ${final_cost:.2f}\n")
        f.write(f"Target (75%+): {'ACHIEVED' if best_acc1 >= 75.0 else 'NOT REACHED'}\n")
        f.write(f"Budget ($15): {'WITHIN' if final_cost <= 15.0 else 'EXCEEDED'}\n")
        f.write(f"Best Model: best_model.pth\n")
        f.write(f"Log File: {log_file}\n")
    
    log_and_print(f"📄 Training summary saved to: {summary_file}")
    log_and_print("="*60)


if __name__ == '__main__':
    main()
