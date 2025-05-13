# Use an official NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:23.04-py3

# Install required system packages
RUN apt-get update && apt-get install -y \
    git \
    curl

# Install vLLM
RUN pip install --upgrade pip
RUN pip install vllm

# Create a working directory
WORKDIR /workspace

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_USE_CUDA_DSA=1

# Command to run the model
CMD ["vllm", "serve", "google/gemma-3-4b-pt"]
