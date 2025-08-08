# Runs vLLM OpenAI-compatible server with GPU and a persistent HF cache
param(
  [string]$Model = "microsoft/Phi-3-mini-4k-instruct",
  [int]$Port = 8000,
  [string]$CacheVolume = "model-cache"
)

docker run --gpus all --rm -p $Port:8000 `
  -v $CacheVolume:/root/.cache/huggingface `
  vllm/vllm-openai:latest-cuda `
  --model $Model
