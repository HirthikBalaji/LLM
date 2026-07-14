import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from .config import LLMConfig
from .layers import (
    RMSNorm,
    SwiGLU,
    GELUMLP,
    DualGranularityEmbedding,
    GatedResidualConnection,
)
from .attention import DifferentialRotaryAttention, StandardMultiHeadAttention


class TransformerBlock(nn.Module):
    """Modular Transformer Decoder Layer."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.mode = config.mode

        # 1. Normalization
        if self.mode in ("diff_llama", "llama"):
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.ln_2 = nn.LayerNorm(config.n_embd)

        # 2. Attention
        if self.mode == "diff_llama":
            self.attn = DifferentialRotaryAttention(config)
        else:
            self.attn = StandardMultiHeadAttention(config)

        # 3. FeedForward Network
        if self.mode in ("diff_llama", "llama"):
            self.mlp = SwiGLU(config.n_embd, dropout=config.dropout)
        else:
            self.mlp = GELUMLP(config.n_embd, dropout=config.dropout)

        # 4. Gated Residual Connections (Novel for diff_llama)
        if self.mode == "diff_llama":
            self.res_attn = GatedResidualConnection(config.n_embd)
            self.res_mlp = GatedResidualConnection(config.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        # Pre-LN Attention
        attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache, use_cache=use_cache)
        if self.mode == "diff_llama":
            x = self.res_attn(x, attn_out)
        else:
            x = x + attn_out

        # Pre-LN MLP
        mlp_out = self.mlp(self.ln_2(x))
        if self.mode == "diff_llama":
            x = self.res_mlp(x, mlp_out)
        else:
            x = x + mlp_out

        return x, new_kv_cache


class LLMTransformer(nn.Module):
    """Main Decoder-only Language Model."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # Token & Position Embeddings
        if config.mode == "diff_llama":
            self.tok_emb = DualGranularityEmbedding(config.vocab_size, config.n_embd, config.dropout)
        else:
            self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        if config.mode == "gpt2":
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
            self.drop = nn.Dropout(config.dropout)
        else:
            self.pos_emb = None

        # Transformer Layers
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # Final LayerNorm
        if config.mode in ("diff_llama", "llama"):
            self.ln_f = RMSNorm(config.n_embd)
        else:
            self.ln_f = nn.LayerNorm(config.n_embd)

        # Language Model Output Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (GPT-2 standard)
        if config.mode != "diff_llama":
            self.lm_head.weight = self.tok_emb.weight if isinstance(self.tok_emb, nn.Embedding) else self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, ...]]]]:
        device = idx.device
        b, t = idx.size()

        # Token embedding
        x = self.tok_emb(idx)

        # Learned position embedding (for GPT-2)
        if self.pos_emb is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.pos_emb(pos)
            x = self.drop(x + pos_emb)

        # Pass through Transformer blocks
        new_kv_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_layer_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_kv_caches.append(new_layer_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive text generation with Temperature, Top-K, Top-P, and KV Cache."""
        self.eval()
        kv_caches = None

        for _ in range(max_new_tokens):
            # If sequence gets longer than block_size and not using cache, crop context
            if not use_cache and idx.size(1) > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size :]
            else:
                idx_cond = idx[:, -1:] if (use_cache and kv_caches is not None) else idx

            logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches, use_cache=use_cache)
            # Take logits of last position
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for batch_idx in range(logits.size(0)):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    logits[batch_idx, indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
