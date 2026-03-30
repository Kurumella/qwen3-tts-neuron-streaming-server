#!/bin/bash
# ==========================================================================
# download_model.sh -- Download Qwen3-TTS model from HuggingFace
#
# Downloads:
#   - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (main model, ~3.4 GB)
#
# Usage:
#   ./download_model.sh                        # download to HF cache
#   ./download_model.sh --output-dir ./model   # download to specific dir
#   ./download_model.sh --check                # check if already downloaded
# ==========================================================================

set -euo pipefail

OUTPUT_DIR=""
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--output-dir /path/to/dir] [--check]"
            echo ""
            echo "Downloads Qwen3-TTS-12Hz-1.7B-CustomVoice from HuggingFace."
            echo ""
            echo "Options:"
            echo "  --output-dir DIR   Download to specific directory (default: HF cache)"
            echo "  --check            Only check if model is already available"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

MODEL_REPO="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# -------------------------------------------------------------------------
# Check if already downloaded
# -------------------------------------------------------------------------

check_model() {
    python3 -c "
import os, sys

# Check HuggingFace cache
cache = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots')
if os.path.exists(cache):
    snapshots = sorted(os.listdir(cache))
    if snapshots:
        path = os.path.join(cache, snapshots[-1])
        config = os.path.join(path, 'config.json')
        if os.path.exists(config):
            print(f'FOUND: {path}')
            sys.exit(0)

# Check home directory
direct = os.path.expanduser('~/Qwen3-TTS-12Hz-1.7B-CustomVoice')
if os.path.exists(direct) and os.path.exists(os.path.join(direct, 'config.json')):
    print(f'FOUND: {direct}')
    sys.exit(0)

print('NOT_FOUND')
sys.exit(1)
" 2>/dev/null
}

echo "=========================================="
echo "  Qwen3-TTS Model Downloader"
echo "=========================================="
echo "  Model: ${MODEL_REPO}"
if [ -n "$OUTPUT_DIR" ]; then
    echo "  Output: ${OUTPUT_DIR}"
else
    echo "  Output: HuggingFace cache (default)"
fi
echo "=========================================="

# Check existing
EXISTING=$(check_model 2>/dev/null || true)
if [[ "$EXISTING" == FOUND:* ]]; then
    MODEL_PATH="${EXISTING#FOUND: }"
    echo ""
    echo "  Model already downloaded: ${MODEL_PATH}"

    if [ "$CHECK_ONLY" = true ]; then
        echo ""
        echo "  Key files:"
        for f in "$MODEL_PATH"/*.safetensors "$MODEL_PATH"/config.json; do
            if [ -e "$f" ]; then
                # Resolve symlinks for real size (HF cache uses blobs)
                REAL=$(readlink -f "$f" 2>/dev/null || echo "$f")
                SIZE=$(du -sh "$REAL" 2>/dev/null | cut -f1)
                echo "    $(basename "$f") (${SIZE})"
            fi
        done
        TOTAL=$(du -shL "$MODEL_PATH" 2>/dev/null | cut -f1)
        echo "  Total (resolved): ${TOTAL}"
        exit 0
    fi

    echo "  Skipping download (already exists)."
    echo ""
    exit 0
fi

if [ "$CHECK_ONLY" = true ]; then
    echo ""
    echo "  Model NOT found. Run without --check to download."
    exit 1
fi

# -------------------------------------------------------------------------
# Download
# -------------------------------------------------------------------------

echo ""
echo "Downloading ${MODEL_REPO}..."
echo "  This may take several minutes (~3.4 GB)."
echo ""

if [ -n "$OUTPUT_DIR" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    '${MODEL_REPO}',
    local_dir='${OUTPUT_DIR}',
)
print(f'Downloaded to: {path}')
"
else
    python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('${MODEL_REPO}')
print(f'Downloaded to: {path}')
"
fi

DOWNLOAD_EXIT=$?
if [ $DOWNLOAD_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Download failed (exit code: $DOWNLOAD_EXIT)"
    echo ""
    echo "Possible fixes:"
    echo "  1. Check internet connectivity"
    echo "  2. Install huggingface_hub: pip install huggingface_hub"
    echo "  3. Login if gated: huggingface-cli login"
    exit 1
fi

# Verify
echo ""
echo "Verifying download..."
VERIFY=$(check_model 2>/dev/null || true)
if [[ "$VERIFY" == FOUND:* ]]; then
    MODEL_PATH="${VERIFY#FOUND: }"
    echo "  Verified: ${MODEL_PATH}"
    echo ""
    echo "  Key files:"
    ls -lh "$MODEL_PATH"/*.safetensors "$MODEL_PATH"/config.json 2>/dev/null | awk '{print "    "$NF" ("$5")"}'
    TOTAL=$(du -sh "$MODEL_PATH" | cut -f1)
    echo "  Total: ${TOTAL}"
else
    echo "  WARNING: Could not verify download location."
fi

echo ""
echo "=========================================="
echo "  Download Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  Native:  ./launch.sh --native --port 8000"
echo "  Docker:  ./build.sh && ./launch.sh --port 8000"
echo ""
