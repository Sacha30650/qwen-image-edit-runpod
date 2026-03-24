FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (bitsandbytes needs CUDA dev headers → devel image)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

CMD ["python3", "handler.py"]
