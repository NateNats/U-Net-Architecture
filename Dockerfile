FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu132

COPY requirements.txt .

RUN pip3 install -r requirements.txt

CMD ["bash"]