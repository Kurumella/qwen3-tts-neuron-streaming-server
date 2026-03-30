"""
OpenAI-compatible streaming TTS server for Qwen3-TTS on NeuronCores.

Endpoint: POST /v1/audio/speech
Compatible with OpenAI's TTS API format.

Architecture:
  - QwenTTSNeuronPipeline with neuronx_distributed TP=2 (both NeuronCores)
  - Thread-safe request queue for concurrent HTTP handling
  - Streaming audio response (chunked PCM/WAV)
  - Single worker thread processes requests sequentially on Neuron hardware

Usage:
  python server.py [--port 8000] [--host 0.0.0.0]
"""

import os
import sys
import io
import time
import wave
import json
import logging
import argparse
import threading
import queue
import uuid
from concurrent.futures import Future

import numpy as np

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, request, Response, jsonify, stream_with_context

logger = logging.getLogger("qwen3-tts-server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE = 24000
MAX_QUEUE_SIZE = 64
MAX_TEXT_LENGTH = 4096
TP_DEGREE = int(os.environ.get("TP_DEGREE", "2"))

# Default model paths
_HF_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots"
)

def _resolve_model_dir():
    """Auto-resolve HuggingFace snapshot directory."""
    env = os.environ.get("TTS_MODEL_DIR")
    if env and os.path.exists(env):
        return env
    if os.path.exists(_HF_CACHE):
        snapshots = sorted(os.listdir(_HF_CACHE))
        if snapshots:
            return os.path.join(_HF_CACHE, snapshots[-1])
    direct = os.path.expanduser("~/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    if os.path.exists(direct):
        return direct
    return None

DEFAULT_MODEL_DIR = _resolve_model_dir()
DEFAULT_TRACE_DIR = os.environ.get(
    "TTS_TRACE_DIR", os.path.join(_PROJECT_ROOT, "neuron_traces")
)

# Qwen3-TTS speaker -> OpenAI voice name mapping
# Users can use either the OpenAI voice names or Qwen3-TTS speaker names
VOICE_TO_SPEAKER = {
    # OpenAI voice names -> Qwen3-TTS speakers
    "alloy": "Ryan",
    "echo": "Aiden",
    "fable": "Serena",
    "onyx": "Eric",
    "nova": "Vivian",
    "shimmer": "Sohee",
}

# Reverse map: allow direct speaker names too
SPEAKER_NAMES = {
    "ryan", "aiden", "serena", "vivian", "eric", "dylan",
    "uncle_fu", "ono_anna", "sohee",
}

# OpenAI language code -> Qwen3-TTS language name
LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "nl": "Dutch",
    "ar": "Arabic",
    "pl": "Polish",
    "tr": "Turkish",
}


def resolve_speaker(voice: str) -> str:
    """Map OpenAI voice name or direct speaker name to Qwen3-TTS speaker."""
    v = voice.strip().lower()
    if v in VOICE_TO_SPEAKER:
        return VOICE_TO_SPEAKER[v]
    # Direct speaker name (case-insensitive)
    for name in SPEAKER_NAMES:
        if v == name:
            # Title-case it (e.g., "ryan" -> "Ryan", "uncle_fu" -> "Uncle_fu")
            return name.capitalize() if "_" not in name else name.title().replace(" ", "")
    # Fallback
    return "Ryan"


def resolve_language(lang: str) -> str:
    """Map language code to Qwen3-TTS language name."""
    lang = lang.strip().lower()
    if lang in LANGUAGE_MAP:
        return LANGUAGE_MAP[lang]
    # Try direct match (already a full name like "English")
    for code, name in LANGUAGE_MAP.items():
        if lang == name.lower():
            return name
    return "English"


# ============================================================================
# Pipeline Manager
# ============================================================================

class PipelineManager:
    """Thread-safe manager for QwenTTSNeuronPipeline.

    TP=2 uses both NeuronCores, so only one inference runs at a time.
    Requests are queued and processed sequentially by a worker thread.
    """

    def __init__(self, model_dir, trace_dir, tp_degree=2, bucket_sizes=None):
        self._pipeline = None
        self._model_dir = model_dir
        self._trace_dir = trace_dir
        self._tp_degree = tp_degree
        self._bucket_sizes = bucket_sizes or [64, 128, 256, 512, 1024]
        self._work_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._worker_thread = None
        self._ready = threading.Event()
        self._shutting_down = False
        self._stats_lock = threading.Lock()
        self._stats = {
            "total_requests": 0,
            "active_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "total_audio_seconds": 0.0,
            "total_inference_seconds": 0.0,
        }

    def start(self):
        """Start worker thread and wait for pipeline to load."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True
        )
        self._worker_thread.start()
        self._ready.wait()

    def _worker_loop(self):
        """Load pipeline, then process queued requests."""
        logger.info("Worker: loading QwenTTSNeuronPipeline (TP=%d)...", self._tp_degree)
        t0 = time.perf_counter()

        from qwen_tts_neuron_pipeline import QwenTTSNeuronPipeline
        self._pipeline = QwenTTSNeuronPipeline(
            model_dir=self._model_dir,
            trace_dir=self._trace_dir,
            bucket_sizes=self._bucket_sizes,
            force_trace=False,
            tp_degree=self._tp_degree,
        )

        elapsed = time.perf_counter() - t0
        logger.info("Worker: pipeline ready in %.1fs", elapsed)
        self._ready.set()

        while not self._shutting_down:
            try:
                work_item = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if work_item is None:
                break

            # Streaming task
            if isinstance(work_item, tuple) and len(work_item) == 2 and work_item[0] == "streaming":
                _, task_fn = work_item
                task_fn()
                continue

            # Regular task
            request_id, params, future = work_item
            try:
                with self._stats_lock:
                    self._stats["active_requests"] += 1
                result = self._run_inference(request_id, params)
                future.set_result(result)
                with self._stats_lock:
                    self._stats["active_requests"] -= 1
                    self._stats["completed_requests"] += 1
            except Exception as e:
                future.set_exception(e)
                with self._stats_lock:
                    self._stats["active_requests"] -= 1
                    self._stats["failed_requests"] += 1

    def _run_inference(self, request_id, params):
        """Execute a single TTS inference."""
        text = params["input"]
        speaker = params["speaker"]
        language = params["language"]

        logger.info("[%s] Generating: speaker=%s lang=%s text=%s...",
                     request_id, speaker, language, text[:80])
        t_start = time.perf_counter()

        wavs, sr = self._pipeline.generate(
            text=text,
            speaker=speaker,
            language=language,
            max_new_tokens=2048,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            non_streaming_mode=True,
        )

        t_end = time.perf_counter()
        inference_time = t_end - t_start

        if sr > 0 and len(wavs) > 0 and len(wavs[0]) > 0:
            wav = wavs[0]
            audio_duration = len(wav) / sr
        else:
            wav = np.zeros(0, dtype=np.float32)
            audio_duration = 0.0

        with self._stats_lock:
            self._stats["total_audio_seconds"] += audio_duration
            self._stats["total_inference_seconds"] += inference_time

        logger.info(
            "[%s] Done: %.2fs wall, %.2fs audio, RTF=%.3f",
            request_id, inference_time, audio_duration,
            inference_time / audio_duration if audio_duration > 0 else float("inf"),
        )

        return {
            "wav": wav,
            "sample_rate": sr,
            "inference_time": inference_time,
            "audio_duration": audio_duration,
            "request_id": request_id,
        }

    def _run_streaming_inference(self, request_id, params):
        """Execute TTS and yield audio chunks incrementally.

        Uses generate_streaming() to get codec tokens as they are produced,
        then decodes each batch to audio and yields immediately for true
        low-latency streaming (TTFA in tens of milliseconds).
        """
        text = params["input"]
        speaker = params["speaker"]
        language = params["language"]

        # Streaming parameters: first chunk is small for low TTFA,
        # subsequent chunks are larger for efficiency.
        first_chunk_tokens = 6   # ~0.5s audio, yielded ASAP
        chunk_tokens = 12        # ~1.0s audio per subsequent chunk
        left_context_size = 10   # context tokens for smooth decode boundaries

        logger.info("[%s] Streaming: speaker=%s lang=%s text=%s...",
                     request_id, speaker, language, text[:80])
        t_start = time.perf_counter()
        t_first_byte = None
        total_audio_samples = 0
        context_codes = []  # Rolling left context for decode

        for codes_batch, ttfa, info in self._pipeline.generate_streaming(
            text=text,
            speaker=speaker,
            language=language,
            max_new_tokens=2048,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            non_streaming_mode=True,
            first_chunk_tokens=first_chunk_tokens,
            chunk_tokens=chunk_tokens,
        ):
            if codes_batch:
                # Decode this chunk with left context for smooth boundaries
                wav_chunk, sr = self._pipeline.decode_codes_chunk(
                    codes_batch,
                    left_context=context_codes[-left_context_size:] if context_codes else None,
                )

                # Update rolling context
                context_codes.extend(codes_batch)

                if len(wav_chunk) > 0:
                    if t_first_byte is None:
                        t_first_byte = time.perf_counter() - t_start
                    total_audio_samples += len(wav_chunk)
                    yield wav_chunk

        t_end = time.perf_counter()
        inference_time = t_end - t_start
        audio_duration = total_audio_samples / SAMPLE_RATE if total_audio_samples > 0 else 0

        with self._stats_lock:
            self._stats["total_audio_seconds"] += audio_duration
            self._stats["total_inference_seconds"] += inference_time
            self._stats["completed_requests"] += 1
            self._stats["active_requests"] -= 1

        logger.info(
            "[%s] Stream done: %.2fs wall, %.2fs audio, TTFA=%.3fs",
            request_id, inference_time, audio_duration,
            t_first_byte or 0.0,
        )

    def submit(self, params):
        """Submit a blocking TTS request. Returns (Future, request_id)."""
        request_id = str(uuid.uuid4())[:8]
        future = Future()

        with self._stats_lock:
            self._stats["total_requests"] += 1

        try:
            self._work_queue.put_nowait((request_id, params, future))
        except queue.Full:
            future.set_exception(
                RuntimeError("Server overloaded: request queue full")
            )

        return future, request_id

    def submit_streaming(self, params):
        """Submit a streaming TTS request. Returns (generator, request_id)."""
        request_id = str(uuid.uuid4())[:8]
        chunk_queue = queue.Queue(maxsize=64)

        with self._stats_lock:
            self._stats["total_requests"] += 1
            self._stats["active_requests"] += 1

        def _worker_task():
            try:
                for chunk in self._run_streaming_inference(request_id, params):
                    chunk_queue.put(("chunk", chunk))
                chunk_queue.put(("done", None))
            except Exception as e:
                chunk_queue.put(("error", str(e)))

        try:
            self._work_queue.put_nowait(("streaming", _worker_task))
        except queue.Full:
            with self._stats_lock:
                self._stats["active_requests"] -= 1
                self._stats["failed_requests"] += 1
            raise RuntimeError("Server overloaded: request queue full")

        def _chunk_generator():
            while True:
                msg_type, data = chunk_queue.get()
                if msg_type == "done":
                    break
                elif msg_type == "error":
                    raise RuntimeError(f"Inference error: {data}")
                elif msg_type == "chunk":
                    yield data

        return _chunk_generator(), request_id

    def get_stats(self):
        with self._stats_lock:
            return dict(self._stats)

    def get_queue_size(self):
        return self._work_queue.qsize()

    def is_ready(self):
        return self._ready.is_set()

    def get_speakers(self):
        if self._pipeline:
            return self._pipeline.get_supported_speakers()
        return []

    def get_languages(self):
        if self._pipeline:
            return self._pipeline.get_supported_languages()
        return []

    def shutdown(self):
        self._shutting_down = True
        self._work_queue.put(None)


# ============================================================================
# Audio Encoding
# ============================================================================

def encode_pcm16(wav_np):
    """float32 [-1,1] -> raw PCM16 s16le bytes (24kHz mono)."""
    pcm16 = (np.clip(wav_np, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm16.tobytes()


def encode_wav(wav_np, sample_rate=SAMPLE_RATE):
    """float32 -> complete WAV file bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm16 = (np.clip(wav_np, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def encode_mp3(wav_np, sample_rate=SAMPLE_RATE):
    """float32 -> MP3 bytes (requires pydub/ffmpeg)."""
    try:
        from pydub import AudioSegment
        pcm16 = (np.clip(wav_np, -1.0, 1.0) * 32767).astype(np.int16)
        audio = AudioSegment(
            data=pcm16.tobytes(),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )
        buf = io.BytesIO()
        audio.export(buf, format="mp3", bitrate="128k")
        return buf.getvalue()
    except ImportError:
        logger.warning("pydub not installed, falling back to WAV")
        return encode_wav(wav_np, sample_rate)


# ============================================================================
# Flask Application
# ============================================================================

app = Flask(__name__)
pipeline_manager: PipelineManager = None


@app.route("/v1/audio/speech", methods=["POST"])
def create_speech():
    """OpenAI-compatible TTS endpoint.

    Request body:
        {
            "model": "tts-1",           // ignored
            "input": "Hello world",      // required: text to synthesize
            "voice": "alloy",            // optional: OpenAI voice or Qwen3-TTS speaker
            "response_format": "pcm",    // optional: pcm, wav, mp3
            "speed": 1.0,               // optional: 0.25-4.0 (reserved, not yet used)
            "stream": true,              // optional: streaming mode
            "language": "en"             // optional: language code or name
        }

    Response:
        Audio data with Content-Type based on response_format.
        Headers: X-Request-Id, X-Queue-Time, X-Inference-Time, X-Audio-Duration
    """
    # Parse request
    try:
        data = request.get_json(force=True)
    except Exception:
        return _error_response("Invalid JSON", "invalid_request_error", 400)

    text = data.get("input", "").strip()
    if not text:
        return _error_response("'input' is required", "invalid_request_error", 400)
    if len(text) > MAX_TEXT_LENGTH:
        return _error_response(
            f"Input too long (max {MAX_TEXT_LENGTH} chars)",
            "invalid_request_error", 400,
        )

    voice = data.get("voice", "alloy")
    response_format = data.get("response_format", "pcm")
    stream = data.get("stream", True)
    language = data.get("language", "en")

    speaker = resolve_speaker(voice)
    lang_name = resolve_language(language)

    # Validate response format
    format_map = {
        "pcm": ("audio/pcm", encode_pcm16),
        "raw": ("audio/pcm", encode_pcm16),
        "wav": ("audio/wav", encode_wav),
        "mp3": ("audio/mpeg", encode_mp3),
    }
    if response_format not in format_map:
        return _error_response(
            f"Unsupported response_format: {response_format}. Use: pcm, wav, mp3",
            "invalid_request_error", 400,
        )
    content_type, encoder = format_map[response_format]

    params = {
        "input": text,
        "speaker": speaker,
        "language": lang_name,
    }

    t_queued = time.perf_counter()

    if stream:
        # Streaming: yield PCM chunks
        try:
            chunk_gen, req_id = pipeline_manager.submit_streaming(params)
        except RuntimeError as e:
            return _error_response(str(e), "server_error", 503)

        def generate():
            for wav_chunk in chunk_gen:
                yield encode_pcm16(wav_chunk)

        headers = {
            "X-Request-Id": req_id,
            "X-Queue-Time": f"{time.perf_counter() - t_queued:.3f}",
            "X-Sample-Rate": str(SAMPLE_RATE),
        }
        return Response(
            stream_with_context(generate()),
            mimetype="audio/pcm",
            headers=headers,
        )
    else:
        # Non-streaming: wait for full result
        future, req_id = pipeline_manager.submit(params)

        try:
            result = future.result(timeout=300)
        except queue.Full:
            return _error_response("Server overloaded", "server_error", 503)
        except Exception as e:
            return _error_response(str(e), "server_error", 500)

        wav = result["wav"]

        if response_format == "wav":
            audio_bytes = encode_wav(wav, result["sample_rate"])
        elif response_format == "mp3":
            audio_bytes = encode_mp3(wav, result["sample_rate"])
        else:
            audio_bytes = encode_pcm16(wav)

        headers = {
            "X-Request-Id": req_id,
            "X-Inference-Time": f"{result['inference_time']:.3f}",
            "X-Audio-Duration": f"{result['audio_duration']:.3f}",
            "X-Queue-Time": f"{time.perf_counter() - t_queued:.3f}",
            "X-Sample-Rate": str(result["sample_rate"]),
        }
        return Response(audio_bytes, mimetype=content_type, headers=headers)


@app.route("/v1/audio/voices", methods=["GET"])
def list_voices():
    """List available voices (OpenAI-compatible + native speakers)."""
    voices = []
    for oai_name, speaker in VOICE_TO_SPEAKER.items():
        voices.append({
            "voice_id": oai_name,
            "name": oai_name,
            "qwen3_speaker": speaker,
        })

    # Also list native speakers not mapped to OpenAI names
    mapped_speakers = set(v.lower() for v in VOICE_TO_SPEAKER.values())
    for speaker in pipeline_manager.get_speakers():
        if speaker.lower() not in mapped_speakers:
            voices.append({
                "voice_id": speaker.lower(),
                "name": speaker,
                "qwen3_speaker": speaker,
            })

    return jsonify({
        "voices": voices,
        "languages": pipeline_manager.get_languages(),
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    stats = pipeline_manager.get_stats()
    return jsonify({
        "status": "ready" if pipeline_manager.is_ready() else "loading",
        "queue_depth": pipeline_manager.get_queue_size(),
        **stats,
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Detailed performance metrics."""
    stats = pipeline_manager.get_stats()
    completed = stats["completed_requests"]
    avg_inference = (
        stats["total_inference_seconds"] / completed if completed > 0 else 0
    )
    avg_audio = (
        stats["total_audio_seconds"] / completed if completed > 0 else 0
    )
    avg_rtf = (
        stats["total_inference_seconds"] / stats["total_audio_seconds"]
        if stats["total_audio_seconds"] > 0 else 0
    )
    return jsonify({
        **stats,
        "queue_depth": pipeline_manager.get_queue_size(),
        "avg_inference_time": round(avg_inference, 3),
        "avg_audio_duration": round(avg_audio, 3),
        "avg_rtf": round(avg_rtf, 3),
    })


@app.route("/", methods=["GET"])
def index():
    """Service info."""
    return jsonify({
        "service": "Qwen3-TTS Streaming Server",
        "version": "1.0",
        "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "accelerator": f"neuronx_distributed TP={TP_DEGREE}",
        "api": "OpenAI-compatible",
        "endpoints": [
            "POST /v1/audio/speech",
            "GET  /v1/audio/voices",
            "GET  /health",
            "GET  /metrics",
        ],
    })


def _error_response(message, error_type, status_code):
    """Return an OpenAI-compatible error response."""
    return jsonify({
        "error": {
            "message": message,
            "type": error_type,
        }
    }), status_code


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS OpenAI-compatible Streaming Server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help="Path to Qwen3-TTS model directory",
    )
    parser.add_argument(
        "--trace-dir", default=DEFAULT_TRACE_DIR,
        help="Path to Neuron trace cache",
    )
    parser.add_argument(
        "--tp-degree", type=int, default=TP_DEGREE,
        help="Tensor parallelism degree (default: 2)",
    )
    parser.add_argument(
        "--buckets", default="64,128,256,512,1024",
        help="Comma-separated talker bucket sizes",
    )
    args = parser.parse_args()

    if not args.model_dir or not os.path.exists(args.model_dir):
        logger.error(
            "Model directory not found: %s\n"
            "Download the model first:\n"
            "  from huggingface_hub import snapshot_download\n"
            "  snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')",
            args.model_dir,
        )
        sys.exit(1)

    bucket_sizes = [int(b) for b in args.buckets.split(",")]

    global pipeline_manager
    pipeline_manager = PipelineManager(
        model_dir=args.model_dir,
        trace_dir=args.trace_dir,
        tp_degree=args.tp_degree,
        bucket_sizes=bucket_sizes,
    )

    logger.info("Starting pipeline initialization...")
    pipeline_manager.start()

    speakers = pipeline_manager.get_speakers()
    languages = pipeline_manager.get_languages()
    logger.info("Available speakers: %s", speakers)
    logger.info("Available languages: %s", languages)
    logger.info(
        "Server starting on %s:%d (TP=%d, buckets=%s)",
        args.host, args.port, args.tp_degree, bucket_sizes,
    )

    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
