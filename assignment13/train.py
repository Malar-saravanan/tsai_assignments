import os
import time
import math
import logging
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import Llama, LlamaConfig
from typing import Tuple

# Hyperparameters
BATCH_SIZE = 4 # Small batch size for safety
BLOCK_SIZE = 512
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LOG_INTERVAL = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'

# Setup logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
print(f"Using device: {DEVICE}")

def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of data for training.
    
    Args:
        data (np.ndarray): The training data (token IDs).
        block_size (int): The context length.
        batch_size (int): The number of sequences in a batch.
        device (str): The device to move tensors to.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Inputs (x) and targets (y).
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def generate(model: Llama, tokenizer: AutoTokenizer, device: str, max_new_tokens: int = 30) -> str:
    """
    Generate text from the model.
    
    Args:
        model (Llama): The trained model.
        tokenizer (AutoTokenizer): The tokenizer for decoding.
        device (str): The device the model is on.
        max_new_tokens (int): Number of tokens to generate.
        
    Returns:
        str: The generated text.
    """
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with BOS or 0
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)
    
    decoded = tokenizer.decode(context[0].tolist())
    model.train()
    return decoded

def train():
    """
    Main training loop.
    Trains the model for 5000 steps, saving a checkpoint at the end.
    """
    # 1. Setup
    torch.manual_seed(1337)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    print("Loading data...")
    with open('input-1.txt', 'r') as f:
        text = f.read()
    
    # Tokenize
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens)}")
    train_data = np.array(tokens)
    
    # 2. Initialize Model
    print("Initializing model...")
    model = Llama.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None # GradScaler for CUDA
    
    # 3. Training Loop (0-5000)
    print("Starting training for 5000 steps...")
    start_time = time.time()
    
    for iter in range(MAX_ITERS + 1): # 0 to 5000
        # Sample batch
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        
        # Forward
        # Mixed precision context
        if DEVICE == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(xb, yb)
            
            # Backward with scaler
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard for MPS/CPU
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        if iter % LOG_INTERVAL == 0:
            dt = time.time() - start_time
            log_msg = f"step {iter}: loss {loss.item():.4f}, time {dt:.2f}s"
            print(log_msg)
            logging.info(log_msg)
            start_time = time.time()

        if iter % EVAL_INTERVAL == 0:
            # Generate
            generated_text = generate(model, tokenizer, DEVICE)
            gen_msg = f"Generated: {generated_text.replace(chr(10), ' ')}"
            print(gen_msg) # Replace newlines for clean log
            logging.info(gen_msg)
            
    # 4. Save Checkpoint
    print("Saving checkpoint_5000.pt...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': iter,
    }
    torch.save(checkpoint, 'checkpoint_5000.pt')
    print("Checkpoint saved.")
    
    # 5. Stop everything (Simulated by returning)
    return

def resume_and_train():
    """
    Resumption logic.
    Loads the checkpoint from step 5000 and trains for 50 more steps.
    """
    print("\n--- Resuming Training ---")
    # 1. Load Tokenizer & Data (Need these again)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    with open('input-1.txt', 'r') as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    train_data = np.array(tokens)
    
    # 2. Load Model & Checkpoint
    print("Loading checkpoint_5000.pt...")
    config = LlamaConfig()
    model = Llama(config)
    checkpoint = torch.load('checkpoint_5000.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_iter = checkpoint['iter']
    print(f"Resumed from step {start_iter}")
    
    # 3. Train for 50 more steps
    print("Training for 50 more steps...")
    model.train()
    
    for iter in range(start_iter + 1, start_iter + 51):
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        log_msg = f"step {iter}: loss {loss.item():.4f}"
        print(log_msg)
        logging.info(log_msg)
            
    print("Training complete.")
    torch.save(model.state_dict(), "final_model.pt")
    print("Final model saved to final_model.pt")

if __name__ == "__main__":
    if not os.path.exists("checkpoint_5000.pt"):
        train()
    
    resume_and_train()
