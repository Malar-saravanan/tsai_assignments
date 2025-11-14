"""
Data Loading Utilities for Language Model Training
"""

import os
import torch
import numpy as np
import tiktoken
from typing import Tuple, Optional
from pathlib import Path


class TextDataset:
    """
    Dataset class for text data with efficient batching
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        sequence_length: int,
        device: str = 'cpu',
        encoding: str = 'gpt2'
    ):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the text file
            batch_size: Number of sequences per batch
            sequence_length: Length of each sequence
            device: Device to place tensors on
            encoding: Tokenizer encoding to use (default: gpt2)
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device
        
        # Load and tokenize the data
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize
        print(f"Tokenizing with {encoding} encoding...")
        enc = tiktoken.get_encoding(encoding)
        self.tokens = torch.tensor(enc.encode(text), dtype=torch.long)
        
        print(f"Loaded {len(self.tokens):,} tokens")
        print(f"Dataset contains {len(self.tokens) / 1e6:.2f}M tokens")
        
        # Calculate dataset statistics
        self.n_batches = len(self.tokens) // (batch_size * sequence_length)
        print(f"1 epoch = {self.n_batches:,} batches")
        
        # State
        self.current_position = 0
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the next batch of data
        
        Returns:
            Tuple of (inputs, targets) tensors
        """
        B, T = self.batch_size, self.sequence_length
        
        # Get the batch
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        
        # Create inputs and targets
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets (shifted by 1)
        
        # Advance position
        self.current_position += B * T
        
        # Reset if we've reached the end
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)
        
        return x, y
    
    def reset(self):
        """Reset the dataset to the beginning"""
        self.current_position = 0
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size"""
        return int(self.tokens.max().item()) + 1


class DataLoaderLite:
    """
    Lightweight data loader for language modeling
    Compatible with the reference implementation
    """
    
    def __init__(self, B: int, T: int, data_path: str = 'data/input.txt'):
        self.B = B
        self.T = T
        
        # Load tokens from disk
        with open(data_path, 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        
        print(f'Loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')
        
        # State
        self.current_position = 0
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets
        
        # Advance position
        self.current_position += B * T
        
        # Reset if needed
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        
        return x, y


def prepare_data(
    data_path: str,
    output_dir: str = 'data',
    encoding: str = 'gpt2',
    train_split: float = 0.9
) -> Tuple[str, str]:
    """
    Prepare and tokenize data for training
    
    Args:
        data_path: Path to raw text file
        output_dir: Directory to save processed data
        encoding: Tokenizer encoding to use
        train_split: Fraction of data to use for training
    
    Returns:
        Tuple of (train_path, val_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize
    print(f"Tokenizing with {encoding} encoding...")
    enc = tiktoken.get_encoding(encoding)
    tokens = enc.encode(text)
    tokens = np.array(tokens, dtype=np.uint16)
    
    # Split into train and validation
    n = len(tokens)
    split_idx = int(n * train_split)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    # Save
    train_path = os.path.join(output_dir, 'train.bin')
    val_path = os.path.join(output_dir, 'val.bin')
    
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    
    print(f"Saved {len(train_tokens):,} training tokens to {train_path}")
    print(f"Saved {len(val_tokens):,} validation tokens to {val_path}")
    
    return train_path, val_path


class BinaryDataLoader:
    """
    Data loader for pre-tokenized binary data
    More efficient for large datasets
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        sequence_length: int,
        device: str = 'cpu'
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device
        
        # Load tokenized data
        print(f"Loading binary data from {data_path}...")
        self.tokens = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        print(f"Loaded {len(self.tokens):,} tokens")
        self.n_batches = len(self.tokens) // (batch_size * sequence_length)
        print(f"1 epoch = {self.n_batches:,} batches")
        
        self.current_position = 0
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.batch_size, self.sequence_length
        
        # Get batch
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        buf = torch.from_numpy(buf.astype(np.int64))
        
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        # Advance
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        
        return x.to(self.device), y.to(self.device)


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoaderLite(B=4, T=32, data_path='data/input.txt')
    x, y = loader.next_batch()
    print(f"Batch shape: {x.shape}")
    print(f"Target shape: {y.shape}")

