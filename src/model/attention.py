import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import LLMConfig
from .layers import RotaryEmbedding


class DifferentialRotaryAttention(nn.Module):
    """
    Novel Differential Multi-Head Attention with RoPE and KV Cache.
    
    Computes two softmax attention maps (A1 - lambda * A2) to cancel 
    out attention noise, eliminate hallucinations, and focus strictly on salient tokens.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.half_head_dim = self.head_dim // 2

        # Double projection for Q and K (Q1, Q2) and (K1, K2)
        self.q_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Learnable scalar lambda parameters for differential attention cancellation map
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_init = config.diff_lambda_init

        self.block_size = config.block_size
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size * 4)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, 2 * C)
        k = self.k_proj(x)  # (B, T, 2 * C)
        v = self.v_proj(x)  # (B, T, C)

        # Reshape for multi-head: (B, n_head, T, 2 * head_dim)
        q = q.view(B, T, self.n_head, 2 * self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, 2 * self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Split Q into Q1 and Q2, K into K1 and K2
        q1, q2 = q.chunk(2, dim=-1)  # (B, n_head, T, head_dim) each
        k1, k2 = k.chunk(2, dim=-1)

        # Apply RoPE on head_dim pairs with position offset if caching
        start_pos = kv_cache[0].size(2) if kv_cache is not None else 0
        cos, sin = self.rope(x, T, start_pos=start_pos)
        # Apply RoPE to q1, q2, k1, k2
        q1 = self.rope.apply_rope(q1, cos, sin)
        q2 = self.rope.apply_rope(q2, cos, sin)
        k1 = self.rope.apply_rope(k1, cos, sin)
        k2 = self.rope.apply_rope(k2, cos, sin)

        # Append to KV cache if specified
        if kv_cache is not None:
            k1_prev, k2_prev, v_prev = kv_cache
            k1 = torch.cat([k1_prev, k1], dim=2)
            k2 = torch.cat([k2_prev, k2], dim=2)
            v = torch.cat([v_prev, v], dim=2)
            if k1.size(2) > self.block_size:
                k1 = k1[:, :, -self.block_size :, :]
                k2 = k2[:, :, -self.block_size :, :]
                v = v[:, :, -self.block_size :, :]

        new_kv_cache = (k1, k2, v) if use_cache else None
        T_k = k1.size(2)

        # Compute scaling factor
        scale = 1.0 / math.sqrt(self.head_dim)

        # Attention Map 1: (B, n_head, T, T_k)
        attn_scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        # Attention Map 2: (B, n_head, T, T_k)
        attn_scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        # Apply causal mask if T > 1
        if T > 1:
            causal_mask = torch.tril(torch.ones(T, T_k, device=x.device)).view(1, 1, T, T_k)
            attn_scores1 = attn_scores1.masked_fill(causal_mask == 0, float("-inf"))
            attn_scores2 = attn_scores2.masked_fill(causal_mask == 0, float("-inf"))

        attn_weights1 = F.softmax(attn_scores1, dim=-1)
        attn_weights2 = F.softmax(attn_scores2, dim=-1)

        # Compute dynamic lambda scalar: exp(q_score * k_score) + lambda_init
        lam = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1) - torch.dot(self.lambda_q2, self.lambda_k2)) + self.lambda_init
        lam = torch.clamp(lam, 0.0, 1.0)

        # Differential Attention combination: A1 - lambda * A2
        diff_attn = attn_weights1 - lam * attn_weights2
        diff_attn = self.attn_dropout(diff_attn)

        # Weighted sum of values
        out = torch.matmul(diff_attn, v)  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        out = self.resid_dropout(self.out_proj(out))
        return out, new_kv_cache


class StandardMultiHeadAttention(nn.Module):
    """Standard Causal Multi-Head Attention with KV Caching."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.block_size = config.block_size
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size * 4)
        self.use_rope = (config.mode == "llama")

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            start_pos = kv_cache[0].size(2) if kv_cache is not None else 0
            cos, sin = self.rope(x, T, start_pos=start_pos)
            q = self.rope.apply_rope(q, cos, sin)
            k = self.rope.apply_rope(k, cos, sin)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
            if k.size(2) > self.block_size:
                k = k[:, :, -self.block_size :, :]
                v = v[:, :, -self.block_size :, :]

        new_kv_cache = (k, v) if use_cache else None
        T_k = k.size(2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if T > 1:
            causal_mask = torch.tril(torch.ones(T, T_k, device=x.device)).view(1, 1, T, T_k)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out, new_kv_cache
