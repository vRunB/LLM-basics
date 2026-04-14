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
