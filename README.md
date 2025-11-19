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
- Run the vLLM OpenAI-compatible server (quick start):
  ```powershell
  .\scripts\run-server.ps1 -Model microsoft/Phi-3-mini-4k-instruct -Port 8000 -CacheVolume model-cache
  ```
- Or build and run this repo’s `app.py` inside a GPU-enabled container:
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

## Docker scripts (Windows PowerShell)

Two helper scripts are provided in `scripts/` to simplify GPU-enabled runs with Docker:

- `scripts/run-server.ps1`
  - Starts the vLLM OpenAI-compatible HTTP server in a container.
  - Parameters:
    - `-Model` (string, default `"microsoft/Phi-3-mini-4k-instruct"`): HF model ID.
    - `-Port` (int, default `8000`): Host port to map to container port 8000.
    - `-CacheVolume` (string, default `"model-cache"`): Named Docker volume for HF cache at `/root/.cache/huggingface` inside the container.
  - Example:
    ```powershell
    .\scripts\run-server.ps1 -Model microsoft/Phi-3-mini-4k-instruct -Port 8000 -CacheVolume model-cache
    ```
  - After it’s up, call `http://localhost:8000` with an OpenAI-compatible client.

- `scripts/run-app.ps1`
  - Builds the Docker image from the included `Dockerfile` and runs `app.py` inside a GPU-enabled container.
  - Parameters:
    - `-Image` (string, default `"demo-vllm"`): Name to tag the built image.
    - `-CacheVolume` (string, default `"model-cache"`): Named Docker volume for HF cache.
  - Example:
    ```powershell
    .\scripts\run-app.ps1 -Image demo-vllm -CacheVolume model-cache
    ```

Notes:
- If PowerShell blocks scripts, allow local scripts once:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```
- The named volume persists downloaded model weights between runs.
- Ensure Docker Desktop is in Linux containers mode and GPU support is enabled.

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
