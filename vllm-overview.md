# vLLM Overview

vLLM is an open-source, high-performance inference engine for large language models (LLMs). It focuses on **fast, efficient serving** of models from the Hugging Face Hub and other sources, especially on GPUs.

## What vLLM Is Good At

- **High-throughput inference**: Designed to serve many requests per second with low latency.
- **GPU-focused performance**: Optimized for NVIDIA GPUs and CUDA-based environments (Linux).
- **Hugging Face integration**: Can load many HF models directly by model ID.
- **OpenAI-compatible server**: Exposes an HTTP API similar to the OpenAI `/v1` endpoints.

## Key Technical Features

- **PagedAttention (efficient KV cache)**  
  Manages the key-value attention cache in a paged manner, reducing memory fragmentation and enabling efficient reuse of GPU memory.

- **Continuous batching**  
  Dynamically groups incoming requests into batches instead of using fixed batch sizes. This keeps the GPU busy and improves both throughput and latency under real-world traffic.

- **Tensor parallelism**  
  Splits large model weights across multiple GPUs so that very large models can be served and scaled horizontally.

- **IO-aware scheduling**  
  Schedules work to balance compute and memory transfers, helping avoid GPU underutilization caused by I/O bottlenecks.

- **OpenAI-style APIs**  
  Provides a server mode (`vllm/vllm-openai` Docker image) that accepts Chat Completions and Completions requests with familiar parameters (e.g., `model`, `messages`, `temperature`, `max_tokens`).

## What Is Attention Cache?

Transformers compute attention over all previous tokens. Naively, each new token would recompute attention from scratch, which is expensive for long sequences. An **attention cache** (often called a **KV cache**) stores the **keys (K)** and **values (V)** for past tokens so they do not need to be recomputed.

When generating token $t$:

- The model only computes Q/K/V for the **new** token.
- It then attends to the **cached** K/V from tokens $1..t-1$.

This turns generation into "one forward pass per new token" instead of "re-run the whole sequence every step", greatly improving speed and reducing redundant work. vLLM's PagedAttention optimizes how this cache is laid out and reused in GPU memory.

## Platform Notes

- vLLM **does not natively support Windows**; it ships CUDA extensions for Linux.  
  On Windows you typically run it via **WSL2** or **Docker with GPU**.
- For environments without GPU or vLLM support, you can still run the same models using the Hugging Face `transformers` library on CPU (slower, but more portable).
