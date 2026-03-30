#!/bin/bash
# ==========================================================================
# build.sh -- Build the Qwen3-TTS Docker image with models and Neuron traces
#
# Two-phase build:
#   Phase 1: docker build  -- installs deps, copies code + model weights
#   Phase 2: docker run    -- traces models on NeuronCores (TP=2), then commits
#
# MUST be run on a trn1 / inf2 instance (requires /dev/neuron0) for Phase 2.
#
# Usage:
#   ./build.sh                                  # uses auto-detected HF cache
#   ./build.sh --download                       # download from HuggingFace first
#   ./build.sh --model-dir /path/to/model       # custom model weights path
#   ./build.sh --no-cache                       # force clean Docker build
#   ./build.sh --skip-trace                     # skip phase 2 (traces on first launch)
#   ./build.sh --buckets 64,128,256,512         # custom bucket sizes
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="${TTS_IMAGE_NAME:-qwen3-tts-neuron-streaming-server}"
IMAGE_TAG="${TTS_IMAGE_TAG:-latest}"
BASE_TAG="${IMAGE_TAG}-base"
TP_DEGREE="${TP_DEGREE:-2}"
BUCKETS="${TTS_BUCKETS:-64,128,256,512,1024}"

MODEL_DIR="${TTS_MODEL_DIR:-}"
SKIP_TRACE=false
DOWNLOAD=false
DOCKER_EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --skip-trace)
            SKIP_TRACE=true
            shift
            ;;
        --download)
            DOWNLOAD=true
            shift
            ;;
        --buckets)
            BUCKETS="$2"
            shift 2
            ;;
        --tp-degree)
            TP_DEGREE="$2"
            shift 2
            ;;
        *)
            DOCKER_EXTRA_ARGS+=("$1")
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

# -------------------------------------------------------------------------
# Download model from HuggingFace if requested
# -------------------------------------------------------------------------

if [ "$DOWNLOAD" = true ]; then
    echo ""
    echo "[Pre-build] Downloading Qwen3-TTS-12Hz-1.7B-CustomVoice from HuggingFace..."
    python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')
print(f'Downloaded to: {path}')
"
fi

# Resolve model directory
MODEL_DIR=$(resolve_model_dir)

# -------------------------------------------------------------------------
# Validate prerequisites
# -------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "  Qwen3-TTS Docker Build"
echo "=========================================="
echo "  Image:      ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Model dir:  ${MODEL_DIR:-NOT FOUND}"
echo "  TP degree:  ${TP_DEGREE}"
echo "  Buckets:    ${BUCKETS}"
echo "  Skip trace: ${SKIP_TRACE}"
echo "=========================================="

if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found."
    echo ""
    echo "Either:"
    echo "  1. Run: ./build.sh --download"
    echo "  2. Use: ./build.sh --model-dir /path/to/model"
    echo "  3. Download manually:"
    echo "     python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')\""
    exit 1
fi

# Check for essential model files
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "ERROR: config.json not found in ${MODEL_DIR}"
    exit 1
fi

SAFETENSOR_COUNT=$(find "$MODEL_DIR" -maxdepth 1 -name "*.safetensors" | wc -l)
if [ "$SAFETENSOR_COUNT" -eq 0 ]; then
    echo "ERROR: No .safetensors files found in ${MODEL_DIR}"
    exit 1
fi

if [ "$SKIP_TRACE" = false ]; then
    if [ ! -e /dev/neuron0 ]; then
        echo "ERROR: /dev/neuron0 not found. Neuron tracing requires a trn1/inf2 instance."
        echo "  Use --skip-trace to build without tracing (traces will be generated on first launch)."
        exit 1
    fi
fi

# -------------------------------------------------------------------------
# Phase 0: Stage model weights into build context
# -------------------------------------------------------------------------

echo ""
echo "[Phase 0] Staging model weights into build context..."

BUILD_MODEL_DIR="${SCRIPT_DIR}/.build_model"

# Clean any previous staging
rm -rf "$BUILD_MODEL_DIR"
mkdir -p "$BUILD_MODEL_DIR"

# Copy all model files (resolve symlinks)
echo "  Copying model files from ${MODEL_DIR}..."
find "$MODEL_DIR" -maxdepth 1 \( -type f -o -type l \) \
    \( -name "*.safetensors" -o -name "*.json" -o -name "*.txt" -o -name "*.model" -o -name "*.tiktoken" -o -name "*.py" \) \
    -exec cp -L {} "$BUILD_MODEL_DIR/" \;

# Also copy speech tokenizer subdirectory if present
if [ -d "${MODEL_DIR}/speech_tokenizer" ]; then
    echo "  Copying speech_tokenizer/..."
    cp -rL "${MODEL_DIR}/speech_tokenizer" "$BUILD_MODEL_DIR/"
fi

MODEL_SIZE=$(du -sh "$BUILD_MODEL_DIR" | cut -f1)
FILE_COUNT=$(find "$BUILD_MODEL_DIR" -type f | wc -l)
echo "  Staged ${MODEL_SIZE} (${FILE_COUNT} files)"

# Stage qwen_tts package (speech tokenizer for codec-to-audio decoding)
BUILD_QWEN_TTS="${SCRIPT_DIR}/.build_qwen_tts"
rm -rf "$BUILD_QWEN_TTS"

QWEN_TTS_SRC="${QWEN_TTS_SRC:-$HOME/Qwen3-TTS/qwen_tts}"
if [ -d "$QWEN_TTS_SRC" ]; then
    echo "  Copying qwen_tts package from ${QWEN_TTS_SRC}..."
    cp -rL "$QWEN_TTS_SRC" "$BUILD_QWEN_TTS"
    # Remove __pycache__ directories
    find "$BUILD_QWEN_TTS" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    # Patch core/__init__.py to make V1 (25Hz) tokenizer import optional
    # (V1 requires 'sox' package which has heavy build-time deps; we only use V2/12Hz)
    cat > "$BUILD_QWEN_TTS/core/__init__.py" << 'PYEOF'
try:
    from .tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Config
    from .tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model
except ImportError:
    Qwen3TTSTokenizerV1Config = None
    Qwen3TTSTokenizerV1Model = None
from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
PYEOF
    QWEN_TTS_SIZE=$(du -sh "$BUILD_QWEN_TTS" | cut -f1)
    echo "  Staged qwen_tts package (${QWEN_TTS_SIZE})"
else
    echo "  WARNING: qwen_tts package not found at ${QWEN_TTS_SRC}"
    echo "           Speech tokenizer will not be available in Docker image."
    echo "           Set QWEN_TTS_SRC=/path/to/qwen_tts to fix."
    mkdir -p "$BUILD_QWEN_TTS"
fi

# -------------------------------------------------------------------------
# Phase 1: Docker build (code + deps + models)
# -------------------------------------------------------------------------

echo ""
echo "[Phase 1] Building base Docker image..."

docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE_NAME}:${BASE_TAG}" \
    "${DOCKER_EXTRA_ARGS[@]+"${DOCKER_EXTRA_ARGS[@]}"}" \
    .

echo "  Base image built: ${IMAGE_NAME}:${BASE_TAG}"

# Clean up staging
rm -rf "$BUILD_MODEL_DIR"
rm -rf "$BUILD_QWEN_TTS"

# -------------------------------------------------------------------------
# Phase 2: Trace models on NeuronCores
# -------------------------------------------------------------------------

if [ "$SKIP_TRACE" = true ]; then
    echo ""
    echo "[Phase 2] SKIPPED (--skip-trace)"
    echo "  Traces will be generated on first server launch (~15-30 min)."
    docker tag "${IMAGE_NAME}:${BASE_TAG}" "${IMAGE_NAME}:${IMAGE_TAG}"
else
    echo ""
    echo "[Phase 2] Tracing models on NeuronCores..."
    echo "  This compiles Talker (TP=${TP_DEGREE}, buckets: ${BUCKETS}) and Code Predictor (TP=${TP_DEGREE})."
    echo "  May take 15-30 minutes depending on instance type and bucket count."

    TRACE_CONTAINER="qwen3-tts-neuron-trace-$$"

    # Clear stale Neuron compile cache to avoid cached failures
    echo "  Clearing Neuron compile cache..."
    rm -rf /var/tmp/neuron-compile-cache 2>/dev/null || true

    # Run tracing inside a container with Neuron device access.
    # We invoke server.py with a special --trace-only flag (it initializes
    # the pipeline which auto-traces, then exits).
    docker run \
        --name "$TRACE_CONTAINER" \
        --device=/dev/neuron0 \
        -e NEURON_RT_NUM_CORES=2 \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e TP_DEGREE="${TP_DEGREE}" \
        -e TTS_PYTHON_CMD="python3 -u" \
        -e NEURON_CC_FLAGS="--retry_failed_compilation" \
        --entrypoint python3 \
        "${IMAGE_NAME}:${BASE_TAG}" \
        -c "
import sys, os
sys.path.insert(0, '/app')
os.environ.setdefault('TTS_MODEL_DIR', '/app/model')
os.environ.setdefault('TTS_TRACE_DIR', '/app/neuron_traces')

from qwen_tts_neuron_pipeline import QwenTTSNeuronPipeline

model_dir = os.environ['TTS_MODEL_DIR']
trace_dir = os.environ['TTS_TRACE_DIR']
tp = int(os.environ.get('TP_DEGREE', '2'))
buckets = [int(b) for b in '${BUCKETS}'.split(',')]

print(f'Tracing: model_dir={model_dir}, trace_dir={trace_dir}, tp={tp}, buckets={buckets}')
pipeline = QwenTTSNeuronPipeline(
    model_dir=model_dir,
    trace_dir=trace_dir,
    bucket_sizes=buckets,
    force_trace=True,
    tp_degree=tp,
)
print('Tracing complete.')
"

    TRACE_EXIT=$?
    if [ $TRACE_EXIT -ne 0 ]; then
        echo "ERROR: Model tracing failed (exit code: $TRACE_EXIT)"
        echo "  Container logs (last 30 lines):"
        docker logs "$TRACE_CONTAINER" 2>&1 | tail -30
        docker rm "$TRACE_CONTAINER" 2>/dev/null
        exit 1
    fi

    # Commit the traced container as the final image
    echo ""
    echo "  Committing traced container as final image..."
    docker commit \
        --change 'ENTRYPOINT ["python3", "server.py"]' \
        --change 'CMD ["--host", "0.0.0.0", "--port", "8000", "--model-dir", "/app/model", "--trace-dir", "/app/neuron_traces"]' \
        --change 'ENV TTS_MODEL_DIR=/app/model' \
        --change 'ENV TTS_TRACE_DIR=/app/neuron_traces' \
        --change 'ENV TTS_PYTHON_CMD="python3 -u"' \
        --change "ENV TP_DEGREE=${TP_DEGREE}" \
        --change 'ENV PATH="/opt/aws/neuron/bin:${PATH}"' \
        --change 'ENV NEURON_RT_NUM_CORES=2' \
        --change 'ENV OMP_NUM_THREADS=1' \
        --change 'ENV MKL_NUM_THREADS=1' \
        --change 'EXPOSE 8000' \
        --change 'WORKDIR /app' \
        "$TRACE_CONTAINER" \
        "${IMAGE_NAME}:${IMAGE_TAG}"

    # Clean up
    docker rm "$TRACE_CONTAINER" 2>/dev/null
    docker rmi "${IMAGE_NAME}:${BASE_TAG}" 2>/dev/null || true

    echo "  Final image committed: ${IMAGE_NAME}:${IMAGE_TAG}"
fi

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "  Build Complete"
echo "=========================================="
IMAGE_SIZE=$(docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" --format='{{.Size}}' 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024/1024}')
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG} (${IMAGE_SIZE})"
if [ "$SKIP_TRACE" = false ]; then
    echo "  Models: baked in (weights + Neuron traces, TP=${TP_DEGREE})"
    echo "  Status: ready to serve immediately"
else
    echo "  Models: weights only (traces generated on first launch)"
fi
echo ""
echo "Launch:"
echo "  ./launch.sh --port 8000"
echo ""
