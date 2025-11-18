# ============================================
# Local LLM Service - Dockerfile
# ============================================
# Build:
#   docker build -t local-llm-service .
#
# Run (CPU):
#   docker run -p 8000:8000 -v ./models:/app/models local-llm-service
#
# Run (GPU):
#   docker run --gpus all -p 8000:8000 -v ./models:/app/models local-llm-service
# ============================================

# Stage 1: Base image
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: CPU version
FROM base as cpu

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p models data models/cache logs

# Expose port
EXPOSE 8000

# Environment variables
ENV USE_GPU=false
ENV GPU_LAYERS=0
ENV CONTEXT_SIZE=4096
ENV MODELS_DIR=/app/models
ENV DATA_DIR=/app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "app.main"]


# Stage 3: GPU version (CUDA)
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 as gpu

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu117

# Install llama-cpp-python with CUDA support
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --no-cache-dir llama-cpp-python --force-reinstall

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p models data models/cache logs

# Expose port
EXPOSE 8000

# Environment variables
ENV USE_GPU=true
ENV GPU_LAYERS=-1
ENV CONTEXT_SIZE=8192
ENV MODELS_DIR=/app/models
ENV DATA_DIR=/app/data
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "app.main"]