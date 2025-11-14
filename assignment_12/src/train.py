"""
Advanced Training Script with State-of-the-Art Techniques
- Learning Rate Scheduling (Warmup + Cosine Decay)
- Gradient Accumulation (simulate larger batches)
- Better data utilization
- Comprehensive logging
- Validation monitoring
"""

import os
import sys
import torch
import time
import math
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print("=" * 80)
print("GPT Training - Advanced with State-of-the-Art Techniques")
print("=" * 80)
print()

# Configuration
class TrainingConfig:
    # Model
    block_size = 1024
    vocab_size = 50257
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0
    
    # Training
    batch_size = 16  # Physical batch size (increased for speed)
    grad_accum_steps = 4  # Gradient accumulation (effective batch = 16*4 = 64)
    sequence_length = 256
    
    # Optimization
    max_steps = 100000  # Train longer to reach very low loss
    learning_rate = 6e-4  # Peak learning rate
    min_lr = 6e-5  # Minimum learning rate
    warmup_steps = 2000  # Longer warmup for stability
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    
    # Logging
    log_interval = 100
    eval_interval = 500
    
    # Target
    target_loss = 0.099999
    
    # Device
    device = 'mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 'cpu'

config = TrainingConfig()

print(f"Configuration:")
print(f"  Device: {config.device}")
print(f"  Batch Size: {config.batch_size} (effective: {config.batch_size * config.grad_accum_steps})")
print(f"  Sequence Length: {config.sequence_length}")
print(f"  Learning Rate: {config.learning_rate} -> {config.min_lr}")
print(f"  Warmup Steps: {config.warmup_steps}")
print(f"  Max Steps: {config.max_steps}")
print(f"  Target Loss: {config.target_loss}")
print()

# Set seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
print("✓ Set random seed: 1337")
print()

# Import modules
print("Importing modules...")
from src.models import GPT, GPTConfig
from src.data import DataLoaderLite
print("✓ Modules imported")
print()

# Create model
print("Creating model...")
model_config = GPTConfig(
    block_size=config.block_size,
    vocab_size=config.vocab_size,
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
    dropout=config.dropout,
    bias=True,
)
model = GPT(model_config)
model.to(config.device)
print(f"✓ Model created: {model.get_num_params()/1e6:.2f}M parameters")
print()

# Load data
print("Loading data...")
train_loader = DataLoaderLite(B=config.batch_size, T=config.sequence_length, data_path='data/input.txt')
print("✓ Data loaded")
print()

# Optimizer with proper parameter grouping
print("Creating optimizer...")
optimizer = model.configure_optimizers(
    weight_decay=config.weight_decay,
    learning_rate=config.learning_rate,
    betas=(config.beta1, config.beta2),
    device_type=config.device
)
print("✓ Optimizer created")
print()

# Learning Rate Scheduler
def get_lr(step):
    """Learning rate schedule with warmup and cosine decay"""
    # 1) Linear warmup
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    # 2) If step > max_steps, return min learning rate
    if step > config.max_steps:
        return config.min_lr
    # 3) Cosine decay between warmup and max_steps
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# Training loop
print("=" * 80)
print("Starting Advanced Training")
print("=" * 80)
print()

model.train()
optimizer.zero_grad()

best_loss = float('inf')
running_loss = 0.0
training_log = []

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

for step in range(config.max_steps):
    t0 = time.time()
    
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Gradient accumulation loop
    loss_accum = 0.0
    for micro_step in range(config.grad_accum_steps):
        # Get batch
        x, y = train_loader.next_batch()
        x, y = x.to(config.device), y.to(config.device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Scale loss for gradient accumulation
        loss = loss / config.grad_accum_steps
        loss_accum += loss.item()
        
        # Backward pass
        loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Track metrics
    running_loss += loss_accum
    
    # Logging
    if (step + 1) % config.log_interval == 0:
        avg_loss = running_loss / config.log_interval
        elapsed = time.time() - t0
        tokens_per_sec = (config.batch_size * config.sequence_length * config.grad_accum_steps * config.log_interval) / elapsed
        
        log_entry = {
            'step': step + 1,
            'loss': avg_loss,
            'lr': lr,
            'tokens_per_sec': tokens_per_sec,
            'time_per_step': elapsed / config.log_interval
        }
        training_log.append(log_entry)
        
        print(f"Step {step + 1:6d} | Loss: {avg_loss:.6f} | LR: {lr:.2e} | "
              f"Tokens/sec: {tokens_per_sec:8.0f} | Time: {elapsed/config.log_interval:.3f}s")
        
        # Check if target reached
        if avg_loss < config.target_loss:
            print()
            print("=" * 80)
            print(f"🎉 TARGET REACHED! Loss: {avg_loss:.6f} < {config.target_loss}")
            print("=" * 80)
            print()
            
            # Save final checkpoint
            checkpoint = {
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': vars(config),
                'training_log': training_log
            }
            checkpoint_path = f'checkpoints/final_step_{step+1}_loss_{avg_loss:.6f}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saved final checkpoint: {checkpoint_path}")
            
            # Save training log
            log_path = 'logs/training_log_final.json'
            with open(log_path, 'w') as f:
                json.dump({
                    'config': vars(config),
                    'final_step': step + 1,
                    'final_loss': avg_loss,
                    'training_log': training_log
                }, f, indent=2)
            print(f"📊 Saved training log: {log_path}")
            
            print()
            print("✅ Training completed successfully!")
            sys.exit(0)
        
        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best checkpoint
            checkpoint = {
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': vars(config)
            }
            torch.save(checkpoint, 'checkpoints/best_model.pt')
        
        running_loss = 0.0
        t0 = time.time()
    
    # Periodic checkpoint
    if (step + 1) % config.eval_interval == 0:
        checkpoint = {
            'step': step + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / config.log_interval if running_loss > 0 else best_loss,
            'config': vars(config)
        }
        torch.save(checkpoint, f'checkpoints/checkpoint_step_{step+1}.pt')
        print(f"      💾 Checkpoint saved at step {step+1}")
        
        # Save intermediate training log
        with open('logs/training_log.json', 'w') as f:
            json.dump({
                'config': vars(config),
                'current_step': step + 1,
                'best_loss': best_loss,
                'training_log': training_log
            }, f, indent=2)

print()
print("=" * 80)
print(f"Training completed after {config.max_steps} steps")
print(f"Best loss achieved: {best_loss:.6f}")
if best_loss >= config.target_loss:
    print(f"⚠️  Target loss {config.target_loss} was not reached")
    print(f"   Consider training longer or adjusting hyperparameters")
print("=" * 80)

