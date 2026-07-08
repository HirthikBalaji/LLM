import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit (LLaMA style MLP)."""

    def __init__(self, in_features: int, hidden_features: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or int(4 * in_features * 2 / 3)
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, in_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GELUMLP(nn.Module):
    """Classic GPT-2 MLP block."""

    def __init__(self, in_features: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(in_features, 4 * in_features)
        self.c_proj = nn.Linear(4 * in_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for 2D Query/Key rotation."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but using [cos, sin] table
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[start_pos : start_pos + seq_len, :], self.sin_cached[start_pos : start_pos + seq_len, :]

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_head, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)


class DualGranularityEmbedding(nn.Module):
    """
    Novel Dual-Granularity Token Embedding.
    Combines direct token embedding with a character-ngram 1D conv fusion.
    """

    def __init__(self, vocab_size: int, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.conv_fusion = nn.Conv1d(n_embd, n_embd, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.tensor(0.5))  # fusion scalar
        self.drop = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (batch_size, seq_len)
        x = self.tok_emb(idx)  # (B, T, C)
        # Conv fusion along sequence dimension for context-aware embedding
        x_conv = self.conv_fusion(x.transpose(1, 2)).transpose(1, 2)
        out = x + self.gamma * torch.tanh(x_conv)
        return self.drop(out)


class GatedResidualConnection(nn.Module):
    """
    Novel Learnable Channel-wise Residual Gate.
    Dynamically scales output of attention or feedforward sublayer.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(n_embd))

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        return x + self.gate * sublayer_out
