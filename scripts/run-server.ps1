# Runs vLLM OpenAI-compatible server with GPU and a persistent HF cache
param(
  [string]$Model = "microsoft/Phi-3-mini-4k-instruct",
  [int]$Port = 8001,
  [string]$CacheVolume = "model-cache"
)

$portMapping = "${Port}:8000"
$volumeMapping = "${CacheVolume}:/root/.cache/huggingface"

docker run --gpus all --rm `
  -p $portMapping `
  -v $volumeMapping `
  vllm/vllm-openai:latest `
  --model $Model