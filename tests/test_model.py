import unittest
import torch
from src.tokenizer import HMGTokenizer
from src.model import LLMConfig, LLMTransformer


class TestLLMSuite(unittest.TestCase):

    def test_tokenizer(self):
        text = "Hello world! This is a test for HMG Tokenizer."
        tokenizer = HMGTokenizer(max_vocab_size=100)
        tokenizer.train_from_text(text)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        self.assertEqual(text, decoded)
        self.assertGreater(tokenizer.vocab_size, 10)

    def test_diff_llama_model_forward(self):
        config = LLMConfig(
            vocab_size=100,
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=32,
            mode="diff_llama",
        )
        model = LLMTransformer(config)
        model.eval()

        b, t = 2, 16
        idx = torch.randint(0, 100, (b, t))
        targets = torch.randint(0, 100, (b, t))

        logits, loss, _ = model(idx, targets=targets)
        self.assertEqual(logits.shape, (b, t, 100))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

    def test_llama_model_forward(self):
        config = LLMConfig(
            vocab_size=100,
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=32,
            mode="llama",
        )
        model = LLMTransformer(config)
        model.eval()

        b, t = 2, 16
        idx = torch.randint(0, 100, (b, t))
        logits, loss, _ = model(idx)
        self.assertEqual(logits.shape, (b, t, 100))
        self.assertIsNone(loss)

    def test_gpt2_model_forward(self):
        config = LLMConfig(
            vocab_size=100,
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=32,
            mode="gpt2",
        )
        model = LLMTransformer(config)
        model.eval()

        b, t = 2, 16
        idx = torch.randint(0, 100, (b, t))
        logits, loss, _ = model(idx)
        self.assertEqual(logits.shape, (b, t, 100))

    def test_generation_with_kv_cache(self):
        config = LLMConfig(
            vocab_size=50,
            n_layer=2,
            n_head=2,
            n_embd=32,
            block_size=16,
            mode="diff_llama",
        )
        model = LLMTransformer(config)
        model.eval()

        prompt = torch.tensor([[1, 2, 3]])
        out = model.generate(prompt, max_new_tokens=5, use_cache=True)
        self.assertEqual(out.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
