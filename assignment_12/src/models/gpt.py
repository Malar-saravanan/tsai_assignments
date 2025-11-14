"""
GPT Model Architecture - Production Grade Implementation
Decoder-only transformer for language modeling
"""

import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """Configuration class for GPT model"""
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    n_layer: int = 12  # number of transformer blocks
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension
    dropout: float = 0.0  # dropout probability
    bias: bool = True  # use bias in linear layers
    
    @property
    def n_params(self) -> int:
        """Calculate total number of parameters"""
        # Embedding parameters
        params = self.vocab_size * self.n_embd  # token embeddings
        params += self.block_size * self.n_embd  # position embeddings
        
        # Transformer blocks
        for _ in range(self.n_layer):
            # Attention
            params += 3 * self.n_embd * self.n_embd  # qkv projection
            params += self.n_embd * self.n_embd  # output projection
            params += 4 * self.n_embd  # biases (if used)
            
            # MLP
            params += self.n_embd * 4 * self.n_embd  # fc
            params += 4 * self.n_embd * self.n_embd  # projection
            params += 5 * self.n_embd  # biases
            
            # Layer norms
            params += 2 * 2 * self.n_embd  # 2 layer norms per block
        
        # Final layer norm
        params += 2 * self.n_embd
        
        # Note: lm_head shares weights with token embedding
        return params


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by number of heads"
        
        # Key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Causal mask to ensure attention only to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
        
        # Flag for scaled initialization
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dimensionality
        
        # Calculate query, key, values for all heads and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape to (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Causal self-attention: (B, n_head, T, head_size) x (B, n_head, head_size, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, n_head, T, T) x (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Flag for scaled initialization
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),  # position embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Forward the GPT model
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # If we are given some desired targets, calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[dict] = None):
        """Load pretrained GPT-2 model weights from HuggingFace"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args), "Only dropout can be overridden"
        
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt: {model_type}")
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        
        print("Forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024   # always 1024 for GPT model checkpoints
        config_args['bias'] = True         # always True for GPT model checkpoints
        
        # Override dropout if specified
        if 'dropout' in override_args:
            print(f"Overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        
        # Create a from-scratch initialized model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer
        
        # Initialize a HuggingFace/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # Copy while ensuring all parameters are aligned
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # OpenAI checkpoints use Conv1D module, we need to transpose
        assert len(sd_keys_hf) == len(sd_keys), f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    
    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # Separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn  # full param name
                
                if pn.endswith('bias'):
                    # All biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # Weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # Weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        
        # Special case: lm_head.weight is tied with transformer.wte.weight
        # Remove lm_head.weight from decay set since it's the same parameter
        if 'lm_head.weight' in decay:
            decay.remove('lm_head.weight')
        
        # Validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"
        
        # Create the PyTorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        # Use fused AdamW if available (more efficient on CUDA)
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"Using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer


if __name__ == "__main__":
    # Test the model
    config = GPTConfig()
    model = GPT(config)
    print(f"Model initialized with {model.get_num_params()/1e6:.2f}M parameters")

