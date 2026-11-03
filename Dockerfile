# ========================================================================
# [RUN GUIDE]
# Dockerfile  -  Python 3.9 + torch 1.12.0 (CUDA 11.3) + transformers 4.22.2
# Build/start with docker-compose.yml:   docker compose up -d --build
# Rebuild only after changing this file or requirements.txt.
# ========================================================================

FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .

RUN pip install --no-cache-dir \
        torch==1.12.0+cu113 torchvision==0.13.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
    && pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir patchelf \
    && find /usr/local/lib/python3.9/site-packages/torch/lib -name "*.so*" \
        -exec patchelf --clear-execstack {} \; || true

CMD ["sleep", "infinity"]
