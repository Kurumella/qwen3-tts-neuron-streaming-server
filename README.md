# Qwen3-TTS Neuron Streaming Server

OpenAI-compatible streaming TTS server running **Qwen3-TTS 12Hz 1.7B CustomVoice** on **AWS Trainium (trn1)** NeuronCores with tensor parallelism (TP=2).

> **Important:** This project is designed exclusively for **AWS Trainium (trn1) and Inferentia2 (inf2) instances** powered by AWS Neuron SDK. It does **not** use NVIDIA CUDA GPUs. All model inference runs on NeuronCores via `neuronx_distributed` and `torch-neuronx`.

## Features

- **OpenAI-compatible API** (`POST /v1/audio/speech`) -- drop-in replacement for OpenAI TTS
- **AWS Neuron native** -- runs on Trainium/Inferentia2 NeuronCores, not CUDA GPUs
- **Streaming audio** -- incremental codec generation with chunked speech decode for ~400ms TTFA
- **Multiple output formats** -- PCM, WAV, and MP3
- **Tensor parallelism** -- model sharded across 2 NeuronCores via `neuronx_distributed`
- **Multiple voices** -- 9 speakers (Ryan, Aiden, Serena, Vivian, Eric, Dylan, Uncle_fu, Ono_anna, Sohee)
- **Multiple languages** -- English, Chinese, Japanese, Korean, German, French, Spanish, and more
- **OpenAI voice mapping** -- `alloy`->Ryan, `echo`->Aiden, `fable`->Serena, `onyx`->Eric, `nova`->Vivian, `shimmer`->Sohee

## Prerequisites

- AWS **trn1.2xlarge** instance (1 Neuron device, 2 NeuronCores) -- this project does **not** support NVIDIA GPUs
- **Neuron SDK 2.x** (`aws-neuronx-runtime-lib`, `aws-neuronx-tools`, `torch-neuronx`, `neuronx_distributed`)
- **Docker** (recommended) or conda environment with PyTorch + Neuron packages

## Getting Started

### Step 1: Clone the repository

```bash
git clone https://github.com/Kurumella/qwen3-tts-neuron-streaming-server.git
cd qwen3-tts-neuron-streaming-server
```

### Step 2: Download the model

```bash
# Download Qwen3-TTS-12Hz-1.7B-CustomVoice from HuggingFace
./download_model.sh

# Or download to a specific directory
./download_model.sh --output-dir ./model

# Check if already downloaded
./download_model.sh --check
```

### Step 3: Build the Docker image

```bash
# Full build with Neuron tracing (recommended, ~30 min)
./build.sh

# Or download model and build in one step
./build.sh --download

# Build without tracing (traces generated on first launch, ~15-30 min)
./build.sh --skip-trace

# Force clean build (no Docker cache)
./build.sh --no-cache
```

The build runs in two phases:

1. **Phase 1** (`docker build`): Installs Neuron SDK, Python deps, copies code + model weights
2. **Phase 2** (`docker run` + `docker commit`): Traces models on NeuronCores (TP=2), commits traced container as final image

Phase 2 requires `/dev/neuron0` (trn1/inf2 instance). Use `--skip-trace` to build on non-Neuron hardware.

### Step 4: Launch the server

```bash
# Docker mode (default)
./launch.sh --port 8000

# With persistent trace cache (survives container restarts)
./launch.sh --port 8000 --trace-dir /data/neuron_traces

# Override model directory
./launch.sh --port 8000 --model-dir /data/model

# Native mode (no Docker, requires conda env or system Python with Neuron packages)
./launch.sh --native --port 8000
```

### Step 5: Test the server

```bash
# Health check
curl -s http://localhost:8000/health | python3 -m json.tool
```

## API Reference

### `POST /v1/audio/speech`

Generate speech from text. Compatible with [OpenAI's TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech).

**Request body:**

```json
{
  "model": "tts-1",
  "input": "Hello, world!",
  "voice": "Ryan",
  "response_format": "mp3",
  "stream": false,
  "language": "en"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | *required* | Text to synthesize (max 4096 chars) |
| `model` | string | -- | Ignored (always uses Qwen3-TTS) |
| `voice` | string | `"Ryan"` | Qwen3-TTS speaker name or OpenAI voice alias |
| `response_format` | string | `"pcm"` | `pcm`, `wav`, or `mp3` |
| `stream` | bool | `true` | Stream audio chunks or wait for complete response |

> **Note:** The defaults are `stream: true` and `response_format: "pcm"`. Streaming mode **always returns raw PCM16 audio** regardless of `response_format`. To get a playable MP3 or WAV file, you **must** set both `"response_format": "mp3"` (or `"wav"`) **and** `"stream": false`.
| `language` | string | `"en"` | Language code (`en`, `zh`, `ja`, `ko`, etc.) or name |
| `speed` | float | `1.0` | Reserved for future use |

**Response (streaming, `stream: true`):**
- Content-Type: `audio/pcm`
- Body: Raw PCM16 s16le chunks (24kHz, mono, 16-bit)
- Headers: `X-Request-Id`, `X-Queue-Time`, `X-Sample-Rate`

**Response (non-streaming, `stream: false`):**
- Content-Type: depends on `response_format` (`audio/pcm`, `audio/wav`, or `audio/mpeg`)
- Body: Complete audio data
- Headers: `X-Request-Id`, `X-Inference-Time`, `X-Audio-Duration`, `X-Sample-Rate`

### `GET /v1/audio/voices`

List available voices and languages.

### `GET /health`

Server health and queue status.

### `GET /metrics`

Performance metrics (average inference time, RTF, throughput).

## Usage Examples

### cURL

```bash
# MP3 output (non-streaming)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test of the Qwen3 text to speech system.",
    "voice": "Ryan",
    "response_format": "mp3",
    "stream": false,
    "language": "en"
  }' \
  --output ryan_test_speech.mp3

# WAV output (non-streaming, Serena voice)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "The quick brown fox jumps over the lazy dog.",
    "voice": "Serena",
    "response_format": "wav",
    "stream": false
  }' \
  --output output.wav

# Raw PCM output (non-streaming, Aiden voice)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Raw PCM audio test.",
    "voice": "Aiden",
    "response_format": "pcm",
    "stream": false
  }' \
  --output output.pcm
# Convert PCM to WAV: ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav

# Streaming PCM (chunked transfer, low latency, Vivian voice)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Streaming audio delivers chunks as they are generated.",
    "voice": "Vivian",
    "stream": true
  }' \
  --output stream_output.pcm

# Chinese language (Eric voice)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "你好，这是一个语音合成测试。",
    "voice": "Eric",
    "response_format": "mp3",
    "stream": false,
    "language": "zh"
  }' \
  --output chinese_test.mp3

# List available voices
curl -s http://localhost:8000/v1/audio/voices | python3 -m json.tool

# Health check
curl -s http://localhost:8000/health | python3 -m json.tool

# Performance metrics
curl -s http://localhost:8000/metrics | python3 -m json.tool
```

### Python (requests)

```python
import requests

# Non-streaming MP3
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": "Ryan",
        "response_format": "mp3",
        "stream": False,
    },
)
with open("output.mp3", "wb") as f:
    f.write(resp.content)

# Streaming PCM
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "Hello world!", "voice": "Serena", "stream": True},
    stream=True,
)
with open("output.pcm", "wb") as f:
    for chunk in resp.iter_content(chunk_size=4096):
        f.write(chunk)
# Convert PCM to WAV: ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.audio.speech.create(
    model="tts-1",
    voice="Ryan",
    input="Hello from Qwen3-TTS on NeuronCores!",
)
response.stream_to_file("output.mp3")
```

## Voice Mapping

| OpenAI Voice | Qwen3-TTS Speaker | Language |
|-------------|-------------------|----------|
| `alloy` | Ryan | English |
| `echo` | Aiden | English |
| `fable` | Serena | Chinese |
| `onyx` | Eric | Chinese |
| `nova` | Vivian | Chinese |
| `shimmer` | Sohee | Korean |

You can also use Qwen3-TTS speaker names directly: `Ryan`, `Aiden`, `Serena`, `Vivian`, `Eric`, `Dylan`, `Uncle_fu`, `Ono_anna`, `Sohee`.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_MODEL_DIR` | Auto-detect HF cache | Path to model weights |
| `TTS_TRACE_DIR` | `./neuron_traces` | Neuron compiled model cache |
| `TP_DEGREE` | `2` | Tensor parallelism degree |
| `TTS_IMAGE_NAME` | `qwen3-tts-neuron-streaming-server` | Docker image name |
| `TTS_IMAGE_TAG` | `latest` | Docker image tag |
| `TTS_PORT` | `8000` | Server port |
| `TTS_BUCKETS` | `64,128,256,512,1024` | Talker bucket sizes |
| `NEURON_RT_NUM_CORES` | `2` | NeuronCores to use |
| `OMP_NUM_THREADS` | `1` | CPU threads (avoid contention) |

### CLI Arguments

```bash
python src/server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model-dir /path/to/model \
  --trace-dir ./neuron_traces \
  --tp-degree 2 \
  --buckets 64,128,256,512,1024
```

## Architecture

```
+-----------------------------------------------------------+
|                      Flask Server                          |
|  POST /v1/audio/speech  GET /health  GET /metrics          |
|                                                            |
|  +----------------+    +------------------------------+    |
|  | HTTP Threads   |--->|    Request Queue (FIFO)      |    |
|  | (concurrent)   |    |    max_size=64               |    |
|  +----------------+    +--------------+---------------+    |
|                                       |                    |
|                        +--------------v---------------+    |
|                        |     Worker Thread (single)    |    |
|                        |  QwenTTSNeuronPipeline TP=2  |    |
|                        +--------------+---------------+    |
+-------------------------------|----------------------------+
                                |
            +-------------------|-----------------------+
            |              Neuron Device                 |
            |  +--------------+  +--------------+       |
            |  | NeuronCore 0 |  | NeuronCore 1 |       |
            |  |  (TP rank 0) |  |  (TP rank 1) |       |
            |  +--------------+  +--------------+       |
            |                                           |
            |  Talker (28 layers)     TP=2 sharded      |
            |  Code Predictor (5 layers) TP=2 sharded   |
            +-------------------------------------------+
```

TP=2 uses both NeuronCores for every inference call. Only one request can execute at a time. The queue pattern lets Flask accept HTTP requests concurrently while the worker processes them sequentially.

### Model Components

| Component | Location | Details |
|-----------|----------|---------|
| **Talker** | NeuronCores (TP=2) | 28 layers, 2048 hidden, 16/8 GQA heads, bucketed [64,128,256,512,1024] |
| **Code Predictor** | NeuronCores (TP=2) | 5 layers, 1024 hidden, 16/8 GQA heads, fixed seq_len=16 |
| **Speech Decoder** | CPU (multi-threaded) | 8-layer transformer + conv upsample, 12Hz->24kHz |
| **Embeddings** | CPU | Text + codec embeddings, projections, sampling |

## Project Structure

```
qwen3-tts-neuron-streaming-server/
├── LICENSE                          # Apache-2.0
├── README.md
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker build definition
├── build.sh                         # Two-phase Docker build (with Neuron tracing)
├── launch.sh                        # Docker / native launch orchestrator
├── download_model.sh                # Download model from HuggingFace
├── .dockerignore
├── src/
│   ├── server.py                    # Flask server with OpenAI-compatible API
│   ├── qwen_tts_neuron_pipeline.py  # Core TTS pipeline (Neuron + CPU)
│   ├── tp_talker.py                 # TP Talker transformer (28 layers)
│   ├── tp_code_predictor.py         # TP Code Predictor (5 layers)
│   ├── trace_tp_talker.py           # Subprocess: trace talker per bucket
│   └── trace_tp_code_predictor.py   # Subprocess: trace code predictor
├── tests/
│   ├── test_server.py               # Functional test suite
│   └── test_concurrency.py          # Performance & concurrency tests
└── neuron_traces/                   # Cached Neuron compiled models (auto-created)
```

## Testing

```bash
# Run functional tests (requires running server)
python tests/test_server.py

# Run performance and concurrency tests
python tests/test_concurrency.py --quick
```

## Performance

Measured on **trn1.2xlarge** (1 Neuron device, 2 NeuronCores, TP=2), Neuron SDK 2.x, BF16 precision, Talker bucket sizes [64, 128, 256, 512, 1024]:

| Metric | Short (5 words) | Medium (16 words) | Long (42 words) |
|--------|-----------------|--------------------|--------------------|
| **TTFA (streaming)** | **400ms** | **401ms** | **407ms** |
| **Inference Time** | 1.45s | 5.96s | 20.21s |
| **Audio Duration** | 1.4s | 6.2s | 19.7s |
| **RTF** | **~1.0x** | **~1.0x** | **~1.0x** |

Concurrency scaling (medium text, streaming):

| Concurrency | Throughput | Latency P50 | Latency P95 |
|-------------|-----------|-------------|-------------|
| 1 | 0.17 req/s | 6.02s | 6.95s |
| 2 | 0.15 req/s | 12.97s | 14.28s |
| 4 | 0.14 req/s | 21.27s | 28.90s |
| 8 | 0.16 req/s | 31.41s | 49.51s |

**Notes:**
- RTF (Real-Time Factor) = audio duration / inference time. RTF of 1.0x means audio is generated at real-time speed.
- TTFA (Time To First Audio) is the time from request to first audio bytes delivered over HTTP. With streaming enabled, the first 6 codec tokens (~0.5s audio) are generated and decoded through the speech decoder, then delivered immediately. Subsequent chunks of 12 tokens (~1.0s audio) follow as they are generated.
- TTFA is approximately constant (~400ms) regardless of text length because it only depends on generating and decoding the first small chunk of codec tokens.
- The Talker (28 layers) and Code Predictor (5 layers) run on NeuronCores (TP=2). The Speech Decoder runs on CPU with multi-threading.
- Throughput is ~0.15-0.17 req/s because TP=2 uses both NeuronCores per request (sequential processing). Concurrency adds queue latency but does not increase per-request throughput.
- First launch includes Neuron trace compilation (~15-30 min). Subsequent launches load from `neuron_traces/` cache in seconds.
- Performance varies with text length, language, speaker, and sampling parameters (temperature, top_k).

## Troubleshooting

### `libfabric.so.1: cannot open shared object file`
Harmless warning during TP tracing. EFA not needed for single-instance TP.

### First launch is slow (15-30 min)
Models must be compiled to Neuron IR on first run. Subsequent launches load from `neuron_traces/` cache.

### `Server not ready` in tests
Wait for pipeline initialization to complete. Check server logs for progress.

### Out of NeuronCore memory
Reduce bucket sizes: `--buckets 64,128,256,512` (drop 1024).

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by QwenLM -- the underlying text-to-speech model this server is built around.
