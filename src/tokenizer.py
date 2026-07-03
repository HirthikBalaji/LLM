import json
import os
from collections import Counter
from typing import List, Dict, Tuple


class HMGTokenizer:
    """
    Hybrid Multi-Granularity (HMG) Tokenizer.
    
    Combines character-level fallback (zero UNK tokens) with dynamic 
    corpus n-gram frequency extraction for efficient subword-style tokenization 
    without external dependencies.
    """

    def __init__(self, max_vocab_size: int = 1000, max_ngram: int = 4):
        self.max_vocab_size = max_vocab_size
        self.max_ngram = max_ngram
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        for idx, token in enumerate(self.special_tokens):
            self.token2id[token] = idx
            self.id2token[idx] = token

    @property
    def pad_id(self) -> int:
        return self.token2id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token2id[self.unk_token]

    @property
    def bos_id(self) -> int:
        return self.token2id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token2id[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def train_from_text(self, text: str):
        """Builds vocabulary using unique characters + top n-grams from corpus."""
        self._init_special_tokens()

        # Step 1: Add all unique characters
        unique_chars = sorted(list(set(text)))
        for ch in unique_chars:
            if ch not in self.token2id:
                idx = len(self.token2id)
                self.token2id[ch] = idx
                self.id2token[idx] = ch

        # Step 2: Extract top n-grams up to max_ngram
        ngram_counts = Counter()
        for n in range(2, self.max_ngram + 1):
            for i in range(len(text) - n + 1):
                ngram = text[i : i + n]
                ngram_counts[ngram] += 1

        # Filter out rare n-grams and add to vocabulary until max_vocab_size
        most_common = ngram_counts.most_common(self.max_vocab_size - len(self.token2id))
        for ngram, _ in most_common:
            if ngram not in self.token2id:
                idx = len(self.token2id)
                self.token2id[ngram] = idx
                self.id2token[idx] = ngram

    def encode(self, text: str) -> List[int]:
        """Encodes text using greedy longest matching n-grams, falling back to chars."""
        tokens = []
        i = 0
        n_text = len(text)

        while i < n_text:
            matched = False
            # Greedy longest match
            for n in range(min(self.max_ngram, n_text - i), 0, -1):
                sub = text[i : i + n]
                if sub in self.token2id:
                    tokens.append(self.token2id[sub])
                    i += n
                    matched = True
                    break
            if not matched:
                tokens.append(self.unk_id)
                i += 1
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decodes token IDs back to a string."""
        chars = []
        for token_id in ids:
            token = self.id2token.get(token_id, "")
            if token not in self.special_tokens:
                chars.append(token)
        return "".join(chars)

    def save_vocab(self, filepath: str):
        """Saves vocabulary dictionary to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "token2id": self.token2id,
            "max_vocab_size": self.max_vocab_size,
            "max_ngram": self.max_ngram,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_vocab(cls, filepath: str) -> "HMGTokenizer":
        """Loads tokenizer from saved JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(
            max_vocab_size=data.get("max_vocab_size", 1000),
            max_ngram=data.get("max_ngram", 4),
        )
        tokenizer.token2id = data["token2id"]
        tokenizer.id2token = {int(v): k for k, v in data["token2id"].items()}
        return tokenizer
