# Standalone Dockerfile for Qwen3-TTS streaming server.
# For the full two-phase build with Neuron tracing, use build.sh instead.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential git wget curl \
    libsndfile1 ffmpeg \
    libarchive13 \
    gnupg2 software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python3*/EXTERNALLY-MANAGED \
    && rm -rf /usr/lib/python3/dist-packages/blinker*

# ---------------------------------------------------------------------------
# Neuron SDK (runtime + tools)
# ---------------------------------------------------------------------------
RUN . /etc/os-release && \
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add - && \
    echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" > /etc/apt/sources.list.d/neuron.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        aws-neuronx-collectives \
        aws-neuronx-runtime-lib \
        aws-neuronx-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies (ordered by change frequency for layer caching)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Neuron Python packages
RUN pip3 install --no-cache-dir \
    neuronx-cc \
    torch-neuronx \
    torchvision \
    neuronx_distributed \
    --extra-index-url https://pip.repos.neuron.amazonaws.com
# torchaudio must match torch-neuronx version and be CPU-only
RUN TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])") && \
    pip3 install --no-cache-dir "torchaudio==${TORCH_VER}" --index-url https://download.pytorch.org/whl/cpu

# HuggingFace dependencies for Qwen3-TTS
# Pin transformers<5 — neuronx_distributed requires transformers.utils.fx (removed in v5)
RUN pip3 install --no-cache-dir "transformers>=4.45,<5" safetensors huggingface_hub

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY src/ ./

# ---------------------------------------------------------------------------
# qwen_tts package -- speech tokenizer for codec-to-audio decoding
# ---------------------------------------------------------------------------
COPY .build_qwen_tts/ ./qwen_tts/

# ---------------------------------------------------------------------------
# Model weights -- baked into the image
# Qwen3-TTS uses HuggingFace snapshot format (safetensors + config)
# ---------------------------------------------------------------------------
COPY .build_model/ ./model/

# Create trace output directory
RUN mkdir -p /app/neuron_traces

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV TTS_MODEL_DIR=/app/model
ENV TTS_TRACE_DIR=/app/neuron_traces
ENV TTS_PYTHON_CMD="python3 -u"
ENV TP_DEGREE=2
ENV PATH="/opt/aws/neuron/bin:${PATH}"
ENV NEURON_RT_NUM_CORES=2
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python3", "server.py"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--model-dir", "/app/model", "--trace-dir", "/app/neuron_traces"]
