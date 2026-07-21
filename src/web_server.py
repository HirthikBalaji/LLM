import argparse
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import torch

from src.tokenizer import HMGTokenizer
from src.model import LLMTransformer
from src.utils import get_device

# Global references for model and tokenizer
model = None
tokenizer = None
device = None
config = None


class LLMWebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        if self.path == "/api/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            info = {
                "model_mode": config.mode if config else "N/A",
                "vocab_size": tokenizer.vocab_size if tokenizer else 0,
                "n_layer": config.n_layer if config else 0,
                "n_head": config.n_head if config else 0,
                "n_embd": config.n_embd if config else 0,
                "params": model.get_num_params() if model else 0,
                "device": str(device),
            }
            self.wfile.write(json.dumps(info).encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode("utf-8"))
            except Exception as e:
                payload = {}

            prompt = payload.get("prompt", "First Citizen:")
            max_new_tokens = int(payload.get("max_new_tokens", 80))
            temperature = float(payload.get("temperature", 0.8))
            top_k = int(payload.get("top_k", 40)) if payload.get("top_k") else None
            top_p = float(payload.get("top_p", 0.9)) if payload.get("top_p") else None

            if model is None or tokenizer is None:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Model or Tokenizer not loaded."}).encode("utf-8"))
                return

            input_ids = tokenizer.encode(prompt)
            if not input_ids:
                input_ids = [tokenizer.bos_id]

            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            out_ids = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=True,
            )

            generated_text = tokenizer.decode(out_ids[0].tolist())

            response = {
                "prompt": prompt,
                "generated_text": generated_text,
                "tokens_generated": len(out_ids[0]) - len(input_ids),
                "model_mode": config.mode,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(checkpoint_path: str, vocab_path: str, port: int = 8000):
    global model, tokenizer, device, config

    device = get_device()
    print(f"Server Device: {device}")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file {vocab_path} not found.")
    tokenizer = HMGTokenizer.load_vocab(vocab_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = LLMTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    server_address = ("", port)
    httpd = HTTPServer(server_address, LLMWebHandler)
    print(f"🚀 LLM Web Playground running at http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Web Playground Server")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--vocab_path", type=str, default="checkpoints/vocab.json")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    run_server(args.checkpoint_path, args.vocab_path, args.port)
