# Builds and runs app.py inside a GPU-enabled container
param(
  [string]$Image = "demo-vllm",
  [string]$CacheVolume = "model-cache"
)

# Build the image
docker build -t $Image .

# Run with GPU and mount a persistent HF cache
docker run --gpus all --rm `
  -v ${PWD}.Path:/app `
  -v ${CacheVolume}:/root/.cache/huggingface `
  -w /app `
  $Image
