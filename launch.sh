#!/bin/bash
# ==========================================================================
# launch.sh -- Launch the Qwen3-TTS streaming server
#
# Modes:
#   1. Docker mode (default if Docker image exists):
#      ./launch.sh --port 8000
#      Models and Neuron traces are baked into the image by build.sh.
#
#   2. Native mode (runs directly with Python):
#      ./launch.sh --native --port 8000
#
# Environment variables:
#   TTS_MODEL_DIR   - path to model weights directory
#   TTS_TRACE_DIR   - path to Neuron trace cache (persistent across restarts)
#   TTS_PORT        - server port (default: 8000)
#   TTS_IMAGE_NAME  - Docker image name (default: qwen3-tts-server)
#   TP_DEGREE       - tensor parallelism degree (default: 2)
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
IMAGE_NAME="${TTS_IMAGE_NAME:-qwen3-tts-neuron-streaming-server}"
IMAGE_TAG="${TTS_IMAGE_TAG:-latest}"
PORT="${TTS_PORT:-8000}"
MODEL_DIR="${TTS_MODEL_DIR:-}"
TRACE_DIR="${TTS_TRACE_DIR:-${SCRIPT_DIR}/neuron_traces}"
TP_DEGREE="${TP_DEGREE:-2}"
BUCKETS="${TTS_BUCKETS:-64,128,256,512,1024}"
NATIVE=false

# Parse arguments
ORIG_ARGS=("$@")
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --native)
            NATIVE=true
            shift
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --trace-dir)
            TRACE_DIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp-degree)
            TP_DEGREE="$2"
            shift 2
            ;;
        --buckets)
            BUCKETS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# -------------------------------------------------------------------------
# Auto-detect model directory from HuggingFace cache
# -------------------------------------------------------------------------

resolve_model_dir() {
    # Check explicit path
    if [ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR" ]; then
        echo "$MODEL_DIR"
        return
    fi

    # Check HuggingFace cache
    local HF_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots"
    if [ -d "$HF_CACHE" ]; then
        local LATEST
        LATEST=$(ls -1 "$HF_CACHE" | sort | tail -1)
        if [ -n "$LATEST" ]; then
            echo "$HF_CACHE/$LATEST"
            return
        fi
    fi

    # Check home directory
    local DIRECT="$HOME/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    if [ -d "$DIRECT" ]; then
        echo "$DIRECT"
        return
    fi

    echo ""
}

MODEL_DIR=$(resolve_model_dir)

echo "=========================================="
echo "  Qwen3-TTS Streaming Server"
echo "=========================================="
echo "  Model dir:  ${MODEL_DIR:-NOT SET (baked into Docker image)}"
echo "  Trace dir:  ${TRACE_DIR}"
echo "  Port:       ${PORT}"
echo "  TP degree:  ${TP_DEGREE}"
echo "  Buckets:    ${BUCKETS}"
echo "=========================================="

if [ "$NATIVE" = true ]; then
    # -------------------------------------------------------------------------
    # Native mode: run directly with Python
    # -------------------------------------------------------------------------

    if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
        echo "ERROR: Model directory not found: ${MODEL_DIR}"
        echo ""
        echo "Either:"
        echo "  1. Download: ./download_model.sh"
        echo "  2. Specify: ./launch.sh --native --model-dir /path/to/model"
        echo "  3. Set TTS_MODEL_DIR environment variable"
        exit 1
    fi

    if [ ! -f "${MODEL_DIR}/config.json" ]; then
        echo "ERROR: config.json not found in ${MODEL_DIR}"
        exit 1
    fi

    echo "  Mode: Native (direct Python)"
    echo "=========================================="

    export NEURON_RT_NUM_CORES=2
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export TTS_MODEL_DIR="$MODEL_DIR"
    export TTS_TRACE_DIR="$TRACE_DIR"
    export TP_DEGREE="$TP_DEGREE"

    # Try conda ldm env first, fallback to system python
    if command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q "^ldm "; then
        echo "  Using conda env: ldm"
        exec conda run --no-capture-output -n ldm \
            python src/server.py \
            --port "$PORT" \
            --model-dir "$MODEL_DIR" \
            --trace-dir "$TRACE_DIR" \
            --tp-degree "$TP_DEGREE" \
            --buckets "$BUCKETS" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    else
        echo "  Using system Python: $(python3 --version 2>&1)"
        exec python3 src/server.py \
            --port "$PORT" \
            --model-dir "$MODEL_DIR" \
            --trace-dir "$TRACE_DIR" \
            --tp-degree "$TP_DEGREE" \
            --buckets "$BUCKETS" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    fi
else
    # -------------------------------------------------------------------------
    # Docker mode
    # -------------------------------------------------------------------------

    echo "  Mode: Docker (${IMAGE_NAME}:${IMAGE_TAG})"
    echo "=========================================="

    # Check if image exists
    if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &>/dev/null; then
        echo "Docker image not found. Building..."
        ./build.sh
    fi

    # Build optional volume mounts
    DOCKER_VOLUMES=()

    # Mount model dir if user explicitly provided one
    if [ -n "${TTS_MODEL_DIR:-}" ] || [[ " ${ORIG_ARGS[*]:-} " == *" --model-dir "* ]]; then
        if [ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR" ]; then
            DOCKER_VOLUMES+=(-v "${MODEL_DIR}:/app/model:ro")
            echo "  Override: mounting external model dir ${MODEL_DIR}"
        fi
    fi

    # Mount trace dir for persistent traces across container restarts
    if [ -n "${TTS_TRACE_DIR:-}" ] || [[ " ${ORIG_ARGS[*]:-} " == *" --trace-dir "* ]]; then
        mkdir -p "$TRACE_DIR"
        DOCKER_VOLUMES+=(-v "${TRACE_DIR}:/app/neuron_traces")
        echo "  Override: mounting external trace dir ${TRACE_DIR}"
    fi

    # Detect TTY for interactive flags
    DOCKER_TTY_FLAGS=()
    if [ -t 0 ]; then
        DOCKER_TTY_FLAGS+=(-it)
    fi

    # Neuron device access (required for inference)
    DOCKER_DEVICE_FLAGS=()
    if [ -e /dev/neuron0 ]; then
        DOCKER_DEVICE_FLAGS+=(--device=/dev/neuron0)
    else
        echo "  WARNING: /dev/neuron0 not found. Container may fail without Neuron hardware."
    fi

    # Run container
    exec docker run --rm \
        "${DOCKER_TTY_FLAGS[@]+"${DOCKER_TTY_FLAGS[@]}"}" \
        --name qwen3-tts-neuron-streaming-server \
        -p "${PORT}:8000" \
        "${DOCKER_DEVICE_FLAGS[@]+"${DOCKER_DEVICE_FLAGS[@]}"}" \
        "${DOCKER_VOLUMES[@]+"${DOCKER_VOLUMES[@]}"}" \
        -e NEURON_RT_NUM_CORES=2 \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e TP_DEGREE="${TP_DEGREE}" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        --host 0.0.0.0 --port 8000 \
        --tp-degree "${TP_DEGREE}" \
        --buckets "${BUCKETS}" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
