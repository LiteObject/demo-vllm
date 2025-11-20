# vLLM Demo

Minimal vLLM demo using `app.py`.

The script loads a Hugging Face text generation model via vLLM and prints a short completion for a given prompt.

vLLM is an open-source, high-performance inference engine for large language models. It achieves high throughput and low latency with techniques like PagedAttention (efficient KV-cache management), continuous batching, tensor parallelism, and IO-aware scheduling. vLLM supports many Hugging Face models, offers a simple Python API and an OpenAI-compatible HTTP server, and is optimized for GPU execution.

## What is Transformers?

Transformers (the Hugging Face `transformers` library) is a popular Python
library for loading, running, and fine-tuning modern Transformer-based models
from the Hugging Face Hub. It provides high-level APIs like
`AutoModelForCausalLM`, `AutoTokenizer`, and `pipeline` for text generation.

In this demo, Transformers is only used as a **CPU fallback** when vLLM is not
available (for example on native Windows). The same model ID can usually be
run through either vLLM (fast GPU inference) or Transformers (portable but
slower CPU inference).

## Windows limitation (important)

vLLM does not natively support Windows. Running `app.py` on Windows typically fails with:

```
ModuleNotFoundError: No module named 'vllm._C'
```

That module is a compiled CUDA extension vLLM only ships for Linux-based platforms. To use this repo with vLLM, run it in a Linux environment (WSL2 or Docker) with an NVIDIA GPU/driver.

## Solutions

You have three practical paths:

1) Run on Linux via WSL2 (recommended on Windows)
- In an elevated PowerShell:
  - `wsl --install -d Ubuntu`
  - Reboot if prompted, then open the Ubuntu app.
- On the Windows side, install the latest NVIDIA GPU driver that supports WSL.
- In Ubuntu (WSL2):
  - Ensure the GPU is visible: `nvidia-smi` (should show your GPU). If not, update drivers.
  - Install Python tools:
    ```bash
    sudo apt update
    sudo apt install -y python3-venv python3-pip
    ```
  - Clone or open this repo inside your Ubuntu home folder (recommended, not the Windows path), then:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install vllm transformers
    python app.py
    ```

2) Docker with NVIDIA GPU
- Install Docker Desktop for Windows and enable the WSL 2 backend.
- Ensure the latest NVIDIA driver is installed on Windows. Enable GPU support in Docker Desktop.
- Verify GPU access:
  ```powershell
  docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
  ```
- Run the vLLM OpenAI-compatible server using docker-compose:
  ```bash
  # Start server (uses settings from .env file)
  docker-compose up
  
  # Or run in background
  docker-compose up -d
  
  # Stop server
  docker-compose down
  ```
- Or build and run this repo's `app.py` inside a GPU-enabled container:
  ```powershell
  .\scripts\run-app.ps1 -Image demo-vllm -CacheVolume model-cache
  ```

3) Transformers CPU fallback (works on native Windows, slower)
- If you just want to test the flow without GPU, you can modify `app.py` to fall back to Hugging Face Transformers when vLLM isn’t available. Example pattern:
  ```python
  try:
      from vllm import LLM, SamplingParams
      USE_VLLM = True
  except (ImportError, RuntimeError):
      USE_VLLM = False

  def demo(model_name: str, prompt: str = "Hello world", max_tokens: int = 50):
      if USE_VLLM:
          llm = LLM(model=model_name)
          outputs = llm.generate([prompt], SamplingParams(max_tokens=max_tokens))
          print("Completion:", outputs[0].outputs[0].text)
      else:
          from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
          tok = AutoTokenizer.from_pretrained(model_name)
          model = AutoModelForCausalLM.from_pretrained(model_name)
          pipe = pipeline("text-generation", model=model, tokenizer=tok)
          out = pipe(prompt, max_new_tokens=max_tokens, do_sample=True)[0]["generated_text"]
          print("Completion:", out)
  ```
- For CPU, use a small model, e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Larger models will be very slow or may OOM on CPU.

## Running vLLM Server with Docker Compose

The easiest way to run the vLLM server is with `docker-compose.yml`.

### Configuration

Edit `.env` file to customize settings:

```env
MODEL=microsoft/Phi-3-mini-4k-instruct
PORT=8001
```

### Commands

```bash
# Start server (foreground)
docker-compose up

# Start server (background/detached)
docker-compose up -d

# Stop server
docker-compose down

# View logs
docker-compose logs -f vllm-server

# Restart with different model
MODEL=meta-llama/Llama-3.2-3B-Instruct docker-compose up
```

### What it does

- Runs `vllm/vllm-openai:latest` with GPU support
- Maps container port 8000 to host port defined in `.env` (default 8001)
- Persists model weights in Docker volume `model-cache`
- Exposes OpenAI-compatible API at `http://localhost:8001`

### Alternative: PowerShell script for app.py

To run `app.py` directly in a container:

```powershell
.\scripts\run-app.ps1 -Image demo-vllm -CacheVolume model-cache
```

Notes:
- The named volume persists downloaded model weights between runs.
- Ensure Docker Desktop is in Linux containers mode and GPU support is enabled.
## Testing the API (server mode)

Once you've started the server with `docker-compose up`, test the OpenAI-compatible endpoints:

```bash
# Chat completions
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "messages": [{"role": "user", "content": "What is vLLM?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Health check
curl http://localhost:8001/health

# List models
curl http://localhost:8001/v1/models
```

Replace `8001` with your chosen port if different.

## Quick start (Linux/WSL2)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install vllm transformers
python app.py
```

## Choosing a model
- Example small-ish instruct models:
  - `microsoft/Phi-3-mini-4k-instruct` (GPU recommended)
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (works on CPU for testing)

Update the call at the bottom of `app.py` to try different models.

## Troubleshooting
- `ModuleNotFoundError: No module named 'vllm._C'` on Windows: use WSL2/Docker or the Transformers CPU fallback as outlined above.
- First run may download model weights from Hugging Face; ensure internet access and sufficient disk space.
- GPU out of memory: pick a smaller model or reduce `max_tokens`.
- Docker: ensure Linux containers and GPU support are enabled in Docker Desktop.

## License
This repo is provided as a simple demo. Follow licenses of vLLM, Transformers, and the chosen model.

