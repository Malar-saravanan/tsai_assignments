"""
Text Generation Script
Generate text using a trained GPT model
"""

import os
import sys
import torch
import argparse
import tiktoken

# Add parent directory to path for imports  
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import GPT, GPTConfig
from src.utils import load_config


def generate_text(
    model: GPT,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = 'cpu'
):
    """
    Generate text from a prompt
    
    Args:
        model: Trained GPT model
        prompt: Text prompt to start generation
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to run on
    
    Returns:
        Generated text
    """
    # Tokenize prompt
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    tokens = tokens.to(device)
    
    # Generate
    model.eval()
    with torch.no_grad():
        generated_tokens = model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_tokens = generated_tokens[0].tolist()
    generated_text = enc.decode(generated_tokens)
    
    return generated_text


def main(args):
    """Main generation function"""
    
    print("=" * 80)
    print("GPT Text Generation")
    print("=" * 80)
    
    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
        device = config['system']['device']
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 'cpu')
    else:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 'cpu')
    
    print(f"\n✓ Using device: {device}")
    
    # Load checkpoint
    print(f"\n📦 Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Reconstruct model configuration
    if 'config' in checkpoint:
        model_cfg = checkpoint['config']
        model_config = GPTConfig(
            block_size=model_cfg.get('model', {}).get('block_size', 1024),
            vocab_size=model_cfg.get('model', {}).get('vocab_size', 50257),
            n_layer=model_cfg.get('model', {}).get('n_layer', 12),
            n_head=model_cfg.get('model', {}).get('n_head', 12),
            n_embd=model_cfg.get('model', {}).get('n_embd', 768),
            dropout=0.0,  # No dropout during inference
            bias=model_cfg.get('model', {}).get('bias', True),
        )
    else:
        # Default GPT-2 config
        model_config = GPTConfig()
    
    # Create and load model
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model with {model.get_num_params() / 1e6:.2f}M parameters")
    print(f"✓ Training step: {checkpoint.get('step', 'unknown')}")
    print(f"✓ Validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    # Generate text
    print("\n" + "=" * 80)
    print("Generating Text")
    print("=" * 80)
    
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = input("\nEnter your prompt: ")
    
    print(f"\nPrompt: {prompt}")
    print("-" * 80)
    
    generated_text = generate_text(
        model=model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    print(generated_text)
    print("-" * 80)
    
    # Interactive mode
    if args.interactive:
        print("\n🔄 Interactive mode (Ctrl+C to exit)")
        try:
            while True:
                prompt = input("\nEnter your prompt: ")
                if not prompt:
                    continue
                
                print("-" * 80)
                generated_text = generate_text(
                    model=model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device
                )
                print(generated_text)
                print("-" * 80)
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with trained GPT model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (optional)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Text prompt for generation'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (higher = more random)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    main(args)

