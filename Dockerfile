FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Pre-download model at build time (baked into image = no cold start download)
RUN python3 -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('Qwen/Qwen-Image-Edit')"

CMD ["python3", "handler.py"]
