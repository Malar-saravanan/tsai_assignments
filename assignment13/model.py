import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LlamaConfig:
    """
    Configuration class for the Llama model.
    
    Attributes:
        block_size (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        n_layer (int): Number of transformer layers.
        n_head (int): Number of query attention heads.
        n_embd (int): Embedding dimension.
        n_kv_head (int): Number of key/value attention heads (for GQA).
        intermediate_size (int): Dimension of the MLP intermediate layer.
        rms_norm_eps (float): Epsilon value for RMSNorm.
        rope_theta (float): Theta value for Rotary Positional Embeddings.
    """
    block_size: int = 2048
    vocab_size: int = 49152
    n_layer: int = 30
    n_head: int = 9
    n_embd: int = 576
    n_kv_head: int = 3
    intermediate_size: int = 1536
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    Args:
        dim (int): The dimension of the input tensor.
        eps (float): A small value to avoid division by zero.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponentials for Rotary Positional Embeddings (RoPE).
    
    Args:
        dim (int): Dimension of the head.
        end (int): Maximum sequence length.
        theta (float): Base for the frequency calculation.
        
    Returns:
        torch.Tensor: Complex tensor of shape (end, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting with input tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Positional Embeddings to query and key tensors.
    
    Args:
        xq (torch.Tensor): Query tensor.
        xk (torch.Tensor): Key tensor.
        freqs_cis (torch.Tensor): Precomputed frequency tensor.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention module with Grouped Query Attention (GQA) and RoPE.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(B, T, self.n_head, self.head_dim)
        xk = xk.view(B, T, self.n_kv_head, self.head_dim)
        xv = xv.view(B, T, self.n_kv_head, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # GQA: repeat k/v heads to match q heads
        if self.n_rep > 1:
            xk = torch.repeat_interleave(xk, self.n_rep, dim=2)
            xv = torch.repeat_interleave(xv, self.n_rep, dim=2)
        
        # Make heads batch dimension
        xq = xq.transpose(1, 2) # (B, n_head, T, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Flash Attention
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(output)

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with SwiGLU activation.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: silu(gate) * up
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    """
    Transformer Block containing Self-Attention and MLP layers with RMSNorm.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Llama(nn.Module):
    """
    Llama Model for Causal Language Modeling.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.embed_tokens.weight = self.lm_head.weight

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head, 
            config.block_size * 2, # Double block size just in case
            config.rope_theta
        )

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Any:
        B, T = idx.size()
        assert T <= self.config.block_size
        
        # Move freqs_cis to device if needed
        if self.freqs_cis.device != idx.device:
            self.freqs_cis = self.freqs_cis.to(idx.device)
            
        freqs_cis = self.freqs_cis[:T]
        
        x = self.embed_tokens(idx)
        
        for layer in self.layers:
            x = layer(x, freqs_cis)
            
        x = self.norm(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :]) # Only last token for generation
            return logits

    @classmethod
    def from_pretrained(cls, model_name: str = "HuggingFaceTB/SmolLM2-135M") -> 'Llama':
        """
        Load pretrained weights from a Hugging Face model.
        
        Args:
            model_name (str): The Hugging Face model ID.
            
        Returns:
            Llama: The initialized model with loaded weights.
        """
        from transformers import AutoModelForCausalLM
        print(f"Loading weights from {model_name}")
        
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        hf_sd = hf_model.state_dict()
        
        config = LlamaConfig()
        model = cls(config)
        sd = model.state_dict()
        
        keys_to_ignore = ["model.rotary_emb.inv_freq"]
        
        for k, v in hf_sd.items():
            if any(ignore in k for ignore in keys_to_ignore):
                continue
                
            # Map HF key to my key
            my_k = k.replace("model.", "")
            if "lm_head" not in k:
                pass
            else:
                my_k = k
            
            if my_k in sd:
                assert sd[my_k].shape == v.shape, f"Shape mismatch for {my_k}: {sd[my_k].shape} vs {v.shape}"
                with torch.no_grad():
                    sd[my_k].copy_(v)
            else:
                print(f"Unmatched key in HF: {k} -> {my_k}")
                
        return model
