#!/usr/bin/env python3
"""
Test suite for the Qwen3-TTS streaming server.

Tests:
  1. Health check and service info
  2. Single request (streaming + non-streaming)
  3. Voice and language support
  4. Error handling (invalid input, missing fields)
  5. Audio validation (PCM, WAV formats)

Usage:
  python test_server.py [--server http://localhost:8000]
  python test_server.py --quick    # Skip long-form tests
"""

import os
import sys
import time
import json
import wave
import struct
import argparse
import io
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SERVER = "http://localhost:8000"
SAMPLE_RATE = 24000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs")


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float = 0.0
    details: str = ""
    error: Optional[str] = None


# ============================================================================
# Utility functions
# ============================================================================

def pcm16_to_duration(data: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    """Calculate audio duration from PCM16 bytes."""
    return (len(data) // 2) / sample_rate


def save_pcm_as_wav(pcm_data: bytes, filepath: str, sample_rate: int = SAMPLE_RATE):
    """Save raw PCM16 bytes as a WAV file."""
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


# ============================================================================
# Test functions
# ============================================================================

def test_service_info(server_url):
    """Test GET / returns service info."""
    t0 = time.perf_counter()
    try:
        resp = requests.get(f"{server_url}/", timeout=10)
        data = resp.json()
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        assert "Qwen3-TTS" in data.get("service", ""), "Missing service name"
        assert "endpoints" in data, "Missing endpoints"
        return TestResult(
            name="GET /",
            passed=True,
            duration=time.perf_counter() - t0,
            details=f"Service: {data['service']}, Model: {data.get('model', 'N/A')}",
        )
    except Exception as e:
        return TestResult(name="GET /", passed=False, error=str(e))


def test_health(server_url):
    """Test GET /health returns ready status."""
    t0 = time.perf_counter()
    try:
        resp = requests.get(f"{server_url}/health", timeout=10)
        data = resp.json()
        assert resp.status_code == 200
        assert data.get("status") == "ready", f"Status: {data.get('status')}"
        return TestResult(
            name="GET /health",
            passed=True,
            duration=time.perf_counter() - t0,
            details=f"Queue: {data.get('queue_depth', 0)}, "
                    f"Completed: {data.get('completed_requests', 0)}",
        )
    except Exception as e:
        return TestResult(name="GET /health", passed=False, error=str(e))


def test_metrics(server_url):
    """Test GET /metrics returns performance data."""
    t0 = time.perf_counter()
    try:
        resp = requests.get(f"{server_url}/metrics", timeout=10)
        data = resp.json()
        assert resp.status_code == 200
        assert "avg_rtf" in data
        return TestResult(
            name="GET /metrics",
            passed=True,
            duration=time.perf_counter() - t0,
            details=f"Avg RTF: {data.get('avg_rtf', 'N/A')}",
        )
    except Exception as e:
        return TestResult(name="GET /metrics", passed=False, error=str(e))


def test_voices(server_url):
    """Test GET /v1/audio/voices returns voice list."""
    t0 = time.perf_counter()
    try:
        resp = requests.get(f"{server_url}/v1/audio/voices", timeout=10)
        data = resp.json()
        assert resp.status_code == 200
        voices = data.get("voices", [])
        assert len(voices) > 0, "No voices returned"
        voice_ids = [v["voice_id"] for v in voices]
        return TestResult(
            name="GET /v1/audio/voices",
            passed=True,
            duration=time.perf_counter() - t0,
            details=f"Voices: {', '.join(voice_ids[:6])}{'...' if len(voice_ids) > 6 else ''}",
        )
    except Exception as e:
        return TestResult(name="GET /v1/audio/voices", passed=False, error=str(e))


def test_streaming_request(server_url, text="Hello, this is a streaming test.",
                           voice="alloy", name_suffix=""):
    """Test streaming POST /v1/audio/speech."""
    test_name = f"Streaming {voice}{name_suffix}"
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{server_url}/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "response_format": "pcm",
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:200]}"

        req_id = resp.headers.get("X-Request-Id", "")
        audio_data = b""
        chunks = 0
        ttfa = None

        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                audio_data += chunk
                chunks += 1

        duration = time.perf_counter() - t0
        audio_dur = pcm16_to_duration(audio_data)

        assert len(audio_data) > 0, "No audio data received"
        assert audio_dur > 0.5, f"Audio too short: {audio_dur:.2f}s"

        # Save output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_pcm_as_wav(
            audio_data,
            os.path.join(OUTPUT_DIR, f"stream_{voice}{name_suffix}.wav"),
        )

        rtf = audio_dur / duration if duration > 0 else 0
        return TestResult(
            name=test_name,
            passed=True,
            duration=duration,
            details=f"Audio: {audio_dur:.2f}s, TTFA: {ttfa*1000:.0f}ms, "
                    f"Chunks: {chunks}, RTF: {rtf:.2f}x, "
                    f"ReqID: {req_id}",
        )
    except Exception as e:
        return TestResult(name=test_name, passed=False, error=str(e),
                         duration=time.perf_counter() - t0)


def test_blocking_request(server_url, text="Hello, this is a blocking test.",
                          voice="alloy", response_format="pcm", name_suffix=""):
    """Test non-streaming POST /v1/audio/speech."""
    test_name = f"Blocking {voice} ({response_format}){name_suffix}"
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{server_url}/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "response_format": response_format,
                "stream": False,
            },
            timeout=120,
        )
        duration = time.perf_counter() - t0

        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:200]}"

        audio_data = resp.content
        assert len(audio_data) > 0, "No audio data"

        req_id = resp.headers.get("X-Request-Id", "")
        inf_time = resp.headers.get("X-Inference-Time", "N/A")
        audio_dur_hdr = resp.headers.get("X-Audio-Duration", "N/A")

        if response_format == "pcm":
            audio_dur = pcm16_to_duration(audio_data)
        elif response_format == "wav":
            buf = io.BytesIO(audio_data)
            with wave.open(buf, "rb") as wf:
                frames = wf.getnframes()
                sr = wf.getframerate()
                audio_dur = frames / sr
        else:
            audio_dur = float(audio_dur_hdr) if audio_dur_hdr != "N/A" else 0

        # Save output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ext = response_format if response_format in ("wav", "mp3") else "wav"
        out_path = os.path.join(OUTPUT_DIR, f"block_{voice}_{response_format}{name_suffix}.{ext}")
        if response_format == "pcm":
            save_pcm_as_wav(audio_data, out_path)
        else:
            with open(out_path, "wb") as f:
                f.write(audio_data)

        return TestResult(
            name=test_name,
            passed=True,
            duration=duration,
            details=f"Audio: {audio_dur:.2f}s, Inference: {inf_time}s, "
                    f"Size: {len(audio_data)//1024}KB, ReqID: {req_id}",
        )
    except Exception as e:
        return TestResult(name=test_name, passed=False, error=str(e),
                         duration=time.perf_counter() - t0)


def test_error_missing_input(server_url):
    """Test error response for missing input."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{server_url}/v1/audio/speech",
            json={"model": "tts-1"},
            timeout=10,
        )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
        data = resp.json()
        assert "error" in data
        return TestResult(
            name="Error: missing input",
            passed=True,
            duration=time.perf_counter() - t0,
            details=f"Error: {data['error']['message']}",
        )
    except Exception as e:
        return TestResult(name="Error: missing input", passed=False, error=str(e))


def test_error_empty_input(server_url):
    """Test error response for empty input."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{server_url}/v1/audio/speech",
            json={"model": "tts-1", "input": ""},
            timeout=10,
        )
        assert resp.status_code == 400
        return TestResult(
            name="Error: empty input",
            passed=True,
            duration=time.perf_counter() - t0,
        )
    except Exception as e:
        return TestResult(name="Error: empty input", passed=False, error=str(e))


def test_error_invalid_json(server_url):
    """Test error response for invalid JSON."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{server_url}/v1/audio/speech",
            data="not json",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert resp.status_code == 400
        return TestResult(
            name="Error: invalid JSON",
            passed=True,
            duration=time.perf_counter() - t0,
        )
    except Exception as e:
        return TestResult(name="Error: invalid JSON", passed=False, error=str(e))


def test_direct_speaker_name(server_url):
    """Test using a Qwen3-TTS speaker name directly instead of OpenAI voice."""
    return test_streaming_request(
        server_url,
        text="Testing direct speaker name access.",
        voice="Aiden",
        name_suffix="_direct_speaker",
    )


def test_long_form(server_url):
    """Test long-form generation."""
    long_text = (
        "Welcome to this comprehensive demonstration of the Qwen three "
        "text to speech system running entirely on AWS Neuron cores. "
        "This technology represents a significant advancement in neural "
        "speech synthesis, enabling real time conversational AI with "
        "remarkably low latency. The system achieves a time to first "
        "audio of approximately ten milliseconds, which means "
        "users experience virtually instant responses."
    )
    return test_blocking_request(
        server_url,
        text=long_text,
        voice="echo",
        response_format="pcm",
        name_suffix="_long",
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Server Test Suite")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Server URL")
    parser.add_argument("--quick", action="store_true", help="Skip long-form tests")
    args = parser.parse_args()

    server = args.server
    results = []

    print(f"\n{'='*70}")
    print(f"  Qwen3-TTS Streaming Server — Test Suite")
    print(f"  Server: {server}")
    print(f"{'='*70}")

    # Phase 1: Connectivity
    print(f"\n--- Phase 1: Connectivity & Info ---")
    for test_fn in [test_service_info, test_health, test_metrics, test_voices]:
        r = test_fn(server)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name} ({r.duration:.3f}s) {r.details or r.error or ''}")

    if not results[1].passed:
        print("\n  Server not ready. Aborting.")
        sys.exit(1)

    # Phase 2: Error Handling
    print(f"\n--- Phase 2: Error Handling ---")
    for test_fn in [test_error_missing_input, test_error_empty_input, test_error_invalid_json]:
        r = test_fn(server)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name} ({r.duration:.3f}s) {r.details or r.error or ''}")

    # Phase 3: Audio Generation
    print(f"\n--- Phase 3: Audio Generation ---")

    # Streaming tests
    r = test_streaming_request(server, voice="alloy")
    results.append(r)
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{status}] {r.name} ({r.duration:.3f}s) {r.details or r.error or ''}")

    r = test_streaming_request(server, voice="echo",
                               text="Hi there! I'm Aiden testing the echo voice.")
    results.append(r)
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{status}] {r.name} ({r.duration:.3f}s) {r.details or r.error or ''}")

    # Blocking tests
    r = test_blocking_request(server, voice="alloy", response_format="pcm")
    results.append(r)
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{status}] {r.name} ({r.duration:.3f}s) {r.details or r.error or ''}")

    r = test_blocking_request(server, voice="alloy", response_format="wav")
    results.append(r)
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{status}] {r.name} ({r.duration:.3f}s) {r.details or r.error or ''}")

    # Direct speaker name
    r = test_direct_speaker_name(server)
    results.append(r)
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{status}] {r.name} ({r.duration:.3f}s) {r.details or r.error or ''}")

    # Long-form
    if not args.quick:
        print(f"\n--- Phase 4: Long-form ---")
        r = test_long_form(server)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name} ({r.duration:.3f}s) {r.details or r.error or ''}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print(f"\n{'='*70}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(results)} total")
    if OUTPUT_DIR and os.path.exists(OUTPUT_DIR):
        print(f"  Audio outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")

    if failed > 0:
        print("\n  Failed tests:")
        for r in results:
            if not r.passed:
                print(f"    - {r.name}: {r.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
