# Simple image to run app.py with GPU-enabled vLLM base
FROM vllm/vllm-openai:latest-cuda

WORKDIR /app

# Copy only what we need
COPY app.py ./

# Extra deps for fallback path (Transformers pipeline)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir transformers

# Default command
CMD ["python", "app.py"]
