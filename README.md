# LLM-basics

A hands-on journey through the evolution of Large Language Models, starting from a foundational Transformer-based GPT and progressing toward modern production-grade architectures like Llama 2.

## Project Overview

This repository contains educational implementations of Generative Pre-trained Transformers (GPT). The goal is to deconstruct the "magic" of LLMs by building them from the ground up, moving from the vanilla Transformer to more optimized, state-of-the-art components.

## Notebooks

### 1. [gpt_from_scratch](https://github.com/vRunB/LLM-basics/edit/main/README.md)
This notebook follows Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out" guide. It implements a decoder-only Transformer based on the "Attention is All You Need" paper, applied to a character-level language modeling task.
* **Key Concepts:**
    * Self-Attention and Multi-Head Attention mechanisms.
    * Positional Encoding (Absolute).
    * Residual connections and Layer Normalization.
    * Bigram-level and Transformer-level training loops.

### 2. [Llama2_Architecture_Evolution:Convert_GPT_2_llama2.ipynb](https://github.com/vRunB/LLM-basics/edit/main/README.md)
This notebook evolves the architecture from the first notebook into a **Llama 2** style model. It swaps out standard components for the more efficient alternatives used by Meta in their flagship open-source model.
* **Key Enhancements:**
    * **RoPE (Rotary Positional Embeddings):** Replaces absolute positional encodings with rotary embeddings to better capture relative positions and improve length extrapolation.
    * **RMSNorm (Root Mean Square Layer Normalization):** A more computationally efficient alternative to standard LayerNorm that simplifies the normalization process.
    * **SwiGLU Activation Function:** (If implemented) Replacing ReLU with SwiGLU for improved training stability and performance.

### 3. [llama2_with_kvcache.ipynb](https://github.com/vRunB/LLM-basics/blob/main/llama2_with_kvcache.ipynb)
This notebook builds on the Llama 2 architecture from the previous notebook and adds **KV (Key-Value) Caching** to speed up autoregressive inference. During generation, the model caches previously computed key and value tensors so that each new token only requires a single forward pass rather than reprocessing the entire sequence.
* **Key Enhancements:**
    * **KV Cache in Attention Heads:** Each `Head` maintains `cache_k` and `cache_v` buffers. During generation, new key/value projections are appended to the cache, and attention is computed against the full cached history.
    * **Position-Aware RoPE:** The `compute_rope` function accepts a `start_pos` parameter so that newly generated tokens receive correct rotary positional encodings relative to their absolute position in the sequence.
    * **Incremental Forward Pass:** The model's `forward` method detects cached token length and only processes uncached tokens, avoiding redundant computation.
    * **Inference Benchmark:** Includes a side-by-side timing comparison of generation with and without the KV cache, demonstrating a measurable speedup (~1.11x on 200 tokens with a 10.7M parameter model).

### 4. [llama2_with_gqa.ipynb](https://github.com/vRunB/LLM-basics/blob/main/llama2_with_gqa.ipynb)
This notebook replaces Multi-Head Attention (MHA) with **Grouped-Query Attention (GQA)**, the technique used by Llama 2 70B, Llama 3, and Mistral. Instead of each query head having its own dedicated Key and Value projections, groups of query heads share a single KV head — reducing KV cache memory and parameter count while preserving model quality.
* **Key Enhancements:**
    * **Grouped KV Projections:** Q is projected to `n_head` heads, while K and V are only projected to `n_kv_head` heads (e.g., 6 query heads share 2 KV heads).
    * **`repeat_interleave` Expansion:** KV heads are expanded at attention time using `repeat_interleave` to match the query head count, keeping the cached tensors compact.
    * **Unified Attention Module:** Replaces the per-head `ModuleList` pattern with a single `GroupedQueryAttention` class that handles all heads in one batched operation.
    * **Parameter Comparison:** Includes a detailed breakdown of MHA vs GQA parameter and KV cache savings (66.7% KV cache reduction with `n_head=6, n_kv_head=2`).

## Getting Started

1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/vRunB/LLM-basics.git
    ```
2.  **Install Dependencies:**
    ```bash
    pip install torch numpy matplotlib
    ```

---
