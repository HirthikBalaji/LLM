# PyTorch Large Language Model (LLM) with Differential Attention & HMG Tokenizer

A modular, high-performance PyTorch implementation of a Decoder-only Large Language Model (LLM) featuring **cutting-edge novelties** in both tokenizer design and neural architecture.

---

## 🌟 Key Novelties & Architecture Highlights

### 1. Tokenizer Novelty: **Hybrid Multi-Granularity (HMG) Tokenizer**
- **Zero-Dependency Subword N-Gram Learning**: Learns frequent character n-grams directly from the corpus based on frequency and mutual information entropy without external libraries.
- **Dual-Granularity Embedding Fusion**: In the embedding layer, token vectors combine subword token embeddings $\mathbf{E}{\text{token}}(t)$ with 1D Convolutional sub-character composition $\text{Conv1D}(\mathbf{E}_{\text{char}}(c_1, c_2, ...))$.
- **0 UNK Guarantee**: Full fallback down to raw characters ensures zero unknown token errors.

### 2. Architecture Novelty: **Differential Rotary Attention (Diff-RoPE Attention)**
- Based on Microsoft's 2024 *Differential Transformer* research:
  $$\mathbf{A}_{\text{diff}} = \text{softmax}\left(\frac{\mathbf{Q}_1 \mathbf{K}_1^T}{\sqrt{d_k}}\right) - \lambda \cdot \text{softmax}\left(\frac{\mathbf{Q}_2 \mathbf{K}_2^T}{\sqrt{d_k}}\right)$$
  where $\lambda$ is a per-head learnable scalar parameter.
- **Why it matters**: Cancels out attention noise and attention sink phenomena, significantly reducing hallucinations and focusing strictly on relevant tokens during autoregressive inference.
- **Rotary Position Embeddings (RoPE)**: Applied dynamically to query and key pairs for relative positioning.

### 3. Architecture Novelty: **Learnable Channel-Wise Residual Gating ($\gamma$-Gating)**
- Dynamic channel-wise learnable gating vectors $\boldsymbol{\gamma}$ scale residual connections ($\mathbf{x} + \boldsymbol{\gamma} \odot \text{SubLayer}(\text{RMSNorm}(\mathbf{x}))$) for smooth gradient flow and depth adaptation.

---

## 📁 Repository Structure

```
.
├── src/
│   ├── tokenizer.py       # Hybrid Multi-Granularity (HMG) Tokenizer
│   ├── dataset.py         # Causal Text Dataset & DataLoader utilities
│   ├── model/
│   │   ├── config.py      # LLMConfig (diff_llama, llama, gpt2 modes)
│   │   ├── layers.py      # RMSNorm, SwiGLU, RoPE, Dual-Embedding, GatedResiduals
│   │   ├── attention.py   # Differential Rotary Attention & Standard MHA with KV Cache
│   │   └── transformer.py # Main LLM Transformer Decoder
│   ├── utils.py           # Device auto-detection (MPS/CUDA/CPU), seeds, checkpoints
│   └── web_server.py      # Zero-dependency Python API server for Web Playground
├── data/
│   └── shakespeare.txt    # Default training corpus
├── checkpoints/           # Trained model weights & saved vocabularies
├── train.py               # Complete training script (Cosine Warmup LR, AMP, AdamW)
├── generate.py            # CLI text generation (Temp, Top-K, Top-P, KV Caching)
├── eval.py                # Perplexity & validation loss evaluation script
├── web/                   # Web Playground UI
│   ├── index.html         # Responsive dark-theme dashboard
│   ├── styles.css         # Glassmorphism design tokens & styles
│   └── app.js             # Real-time generation controls & API handler
├── tests/
│   └── test_model.py      # Unit test suite
├── requirements.txt       # Dependencies (PyTorch)
└── README.md
```

---

## ⚡ Quick Start Guide

### 1. Installation
Ensure PyTorch 2.0+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
Verify model components, differential attention forward pass, and KV caching equivalence:
```bash
python3 -m unittest discover -s tests
```

### 3. Train the LLM
Train the model on the sample dataset with automatic hardware detection (`mps` on Apple Silicon, `cuda`, or `cpu`):
```bash
python3 train.py --epochs 10 --batch_size 16 --mode diff_llama --n_layer 4 --n_head 4 --n_embd 256
```

### 4. Generate Text via CLI
Run autoregressive text completion with sampling controls (Temperature, Top-K, Top-P):
```bash
python3 generate.py --prompt "First Citizen:\nBefore we proceed" --temperature 0.8 --top_k 40
```

### 5. Evaluate Perplexity
Calculate validation loss and perplexity:
```bash
python3 eval.py
```

### 6. Launch Interactive Web Playground
Launch the local web server:
```bash
python3 src/web_server.py --port 8000
```
Open **`http://localhost:8000`** in your browser to interact with the model via a sleek dark-theme playground UI!

---

## 🛠️ Model Modes

The architecture supports three preset modes via `LLMConfig(mode=...)`:
- `diff_llama` (**Recommended Novel Mode**): Differential Attention + RoPE + RMSNorm + SwiGLU + HMG Dual Embedding + Gated Residuals.
- `llama`: Standard LLaMA-style decoder (MHA + RoPE + RMSNorm + SwiGLU).
- `gpt2`: Classic GPT-2 style decoder (MHA + Absolute Learned Positional Embeddings + LayerNorm + GELU).
