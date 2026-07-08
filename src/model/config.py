from dataclasses import dataclass
from typing import Literal


@dataclass
class LLMConfig:
    vocab_size: int = 1000
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.1
    mode: Literal["diff_llama", "llama", "gpt2"] = "diff_llama"
    bias: bool = False  # True for GPT-2 style bias in linears
    
    # Differential Attention specific
    diff_lambda_init: float = 0.8  # Initial lambda for differential attention cancellation map

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
