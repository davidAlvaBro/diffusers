# CUDA runtime base (adjust tag to match your framework's CUDA expectation)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Keep apt non-interactive; unbuffer Python logs
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# System deps + Python 3.11 (no --no-install-recommends as requested)
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-venv \
    ffmpeg libsm6 libxext6 libglm-dev \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

# Make 'python' and 'pip' point to Python 3.10 tools
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Fixed internal paths for where you'll mount things at runtime
ENV CODE_DIR=/app
ENV DATA_DIR=/data
WORKDIR ${CODE_DIR}

COPY . ${CODE_DIR}

RUN python3.11 -m pip install --upgrade pip \
 && python3.11 -m pip install "torch>=2.5.1" "torchvision>=0.20.1" --index-url https://download.pytorch.org/whl/cu124 \
 && python3.11 -m pip install --upgrade transformers accelerate safetensors opencv-python numpy \
 && python3.11 -m pip install -e third_party/diffusers

# Default to a shell
CMD ["bash"]
