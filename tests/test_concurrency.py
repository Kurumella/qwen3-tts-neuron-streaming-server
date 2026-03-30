#!/usr/bin/env python3
"""
Concurrency and performance test suite for the Qwen3-TTS streaming server.

Tests:
  1. Single request baseline (TTFA, inference time, RTF)
  2. Streaming vs non-streaming comparison
  3. Concurrency scaling (2, 4, 8, 16 clients)
  4. Stress test with sustained load

Usage:
  python test_concurrency.py [--server http://localhost:8000]
  python test_concurrency.py --quick --max-concurrency 8
"""

import os
import sys
import time
import json
import argparse
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SERVER = "http://localhost:8000"
SAMPLE_RATE = 24000

TEST_TEXTS = {
    "short": "Hello, this is a test.",
    "medium": (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence tests the full pipeline performance."
    ),
    "long": (
        "Artificial intelligence has transformed the way we interact with "
        "technology. From voice assistants to autonomous vehicles, the "
        "applications are vast and growing. Neural networks continue to "
        "push the boundaries of what machines can achieve, enabling "
        "breakthroughs in healthcare, science, and creative arts."
    ),
}


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class RequestMetrics:
    text_length: int = 0
    ttfa: float = 0.0
    total_latency: float = 0.0
    queue_time: float = 0.0
    inference_time: float = 0.0
    audio_duration: float = 0.0
    audio_bytes: int = 0
    rtf: float = 0.0
    status_code: int = 0
    error: Optional[str] = None
    streaming: bool = False
    chunks_received: int = 0


@dataclass
class TestResult:
    test_name: str
    concurrency: int
    num_requests: int
    metrics: List[RequestMetrics] = field(default_factory=list)
    wall_time: float = 0.0

    @property
    def successful(self):
        return [m for m in self.metrics if m.error is None and m.status_code == 200]

    @property
    def failed(self):
        return [m for m in self.metrics if m.error is not None or m.status_code != 200]

    def summary(self):
        ok = self.successful
        if not ok:
            return {"error": "No successful requests"}

        latencies = [m.total_latency for m in ok]
        ttfas = [m.ttfa for m in ok if m.ttfa > 0]
        rtfs = [m.rtf for m in ok if m.rtf > 0]
        inf_times = [m.inference_time for m in ok if m.inference_time > 0]

        def pcts(data):
            if not data:
                return {}
            s = sorted(data)
            n = len(s)
            return {
                "min": round(s[0], 3),
                "p50": round(s[n // 2], 3),
                "p95": round(s[int(n * 0.95)], 3),
                "max": round(s[-1], 3),
                "mean": round(statistics.mean(s), 3),
            }

        return {
            "test_name": self.test_name,
            "concurrency": self.concurrency,
            "total_requests": self.num_requests,
            "successful": len(ok),
            "failed": len(self.failed),
            "wall_time": round(self.wall_time, 3),
            "throughput_rps": round(len(ok) / self.wall_time, 2) if self.wall_time > 0 else 0,
            "latency": pcts(latencies),
            "ttfa": pcts(ttfas) if ttfas else "N/A",
            "inference_time": pcts(inf_times) if inf_times else "N/A",
            "rtf": pcts(rtfs) if rtfs else "N/A",
        }


# ============================================================================
# Request functions
# ============================================================================

def make_streaming_request(server_url, text, voice="alloy"):
    """Streaming TTS request with metrics."""
    m = RequestMetrics(text_length=len(text), streaming=True)
    t_start = time.perf_counter()
    first_byte_time = None

    try:
        resp = requests.post(
            f"{server_url}/v1/audio/speech",
            json={"model": "tts-1", "input": text, "voice": voice,
                  "response_format": "pcm", "stream": True},
            stream=True, timeout=120,
        )
        m.status_code = resp.status_code
        if resp.status_code != 200:
            m.error = f"HTTP {resp.status_code}"
            m.total_latency = time.perf_counter() - t_start
            return m

        total_bytes = 0
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                if first_byte_time is None:
                    first_byte_time = time.perf_counter()
                total_bytes += len(chunk)
                m.chunks_received += 1

        m.audio_bytes = total_bytes
    except Exception as e:
        m.error = str(e)
        m.total_latency = time.perf_counter() - t_start
        return m

    t_end = time.perf_counter()
    m.total_latency = t_end - t_start
    m.ttfa = (first_byte_time - t_start) if first_byte_time else m.total_latency
    m.audio_duration = (m.audio_bytes // 2) / SAMPLE_RATE
    m.inference_time = m.total_latency  # streaming: total latency is the inference time
    m.rtf = m.audio_duration / m.total_latency if m.total_latency > 0 else 0
    qt = resp.headers.get("X-Queue-Time")
    if qt:
        m.queue_time = float(qt)
    return m


def make_blocking_request(server_url, text, voice="alloy"):
    """Non-streaming TTS request with metrics."""
    m = RequestMetrics(text_length=len(text), streaming=False)
    t_start = time.perf_counter()

    try:
        resp = requests.post(
            f"{server_url}/v1/audio/speech",
            json={"model": "tts-1", "input": text, "voice": voice,
                  "response_format": "pcm", "stream": False},
            timeout=120,
        )
        m.status_code = resp.status_code
        if resp.status_code != 200:
            m.error = f"HTTP {resp.status_code}"
            m.total_latency = time.perf_counter() - t_start
            return m
        m.audio_bytes = len(resp.content)
    except Exception as e:
        m.error = str(e)
        m.total_latency = time.perf_counter() - t_start
        return m

    m.total_latency = time.perf_counter() - t_start
    m.ttfa = m.total_latency
    m.audio_duration = (m.audio_bytes // 2) / SAMPLE_RATE
    it = resp.headers.get("X-Inference-Time")
    if it:
        m.inference_time = float(it)
    else:
        m.inference_time = m.total_latency
    m.rtf = m.audio_duration / m.inference_time if m.inference_time > 0 else 0
    qt = resp.headers.get("X-Queue-Time")
    if qt:
        m.queue_time = float(qt)
    return m


# ============================================================================
# Test runners
# ============================================================================

def test_health(server_url):
    print(f"\n{'='*70}")
    print(f"  Server: {server_url}")
    print(f"{'='*70}")
    try:
        resp = requests.get(f"{server_url}/health", timeout=10)
        data = resp.json()
        print(f"  Status: {data.get('status')}")
        return data.get("status") == "ready"
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def run_single_test(server_url, text_key="medium"):
    text = TEST_TEXTS[text_key]
    print(f"\n{'='*70}")
    print(f"  Single Request Baseline ({text_key}, {len(text)} chars)")
    print(f"{'='*70}")

    # Warmup
    print("  Warmup...")
    make_streaming_request(server_url, "warmup")

    results = []
    for i in range(3):
        m = make_streaming_request(server_url, text)
        results.append(m)
        ok = "OK" if not m.error else f"FAIL: {m.error}"
        print(
            f"  Run {i+1}: TTFA={m.ttfa*1000:.0f}ms  InfTime={m.inference_time:.2f}s  "
            f"Audio={m.audio_duration:.2f}s  RTF={m.rtf:.2f}x  [{ok}]"
        )

    ok = [r for r in results if not r.error]
    if ok:
        print(
            f"\n  Avg: TTFA={statistics.mean(r.ttfa for r in ok)*1000:.0f}ms  "
            f"InfTime={statistics.mean(r.inference_time for r in ok):.2f}s  "
            f"Audio={statistics.mean(r.audio_duration for r in ok):.2f}s  "
            f"RTF={statistics.mean(r.rtf for r in ok):.2f}x"
        )

    return TestResult(
        test_name=f"single_{text_key}", concurrency=1,
        num_requests=len(results), metrics=results,
        wall_time=sum(r.total_latency for r in results),
    )


def run_concurrent_test(server_url, concurrency, num_requests=None,
                        text_key="medium", streaming=True):
    if num_requests is None:
        num_requests = concurrency

    text = TEST_TEXTS[text_key]
    mode = "stream" if streaming else "block"
    print(f"\n{'='*70}")
    print(f"  Concurrency={concurrency}, Requests={num_requests} ({mode})")
    print(f"{'='*70}")

    fn = make_streaming_request if streaming else make_blocking_request
    all_m = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(fn, server_url, text) for _ in range(num_requests)]
        for i, f in enumerate(futures):
            m = f.result()
            all_m.append(m)
            ok = "OK" if not m.error else f"FAIL"
            print(
                f"  Req {i+1:3d}: Lat={m.total_latency:.3f}s  "
                f"Audio={m.audio_duration:.2f}s  [{ok}]"
            )

    wall = time.perf_counter() - t0
    result = TestResult(
        test_name=f"conc_{concurrency}_{mode}",
        concurrency=concurrency, num_requests=num_requests,
        metrics=all_m, wall_time=wall,
    )
    s = result.summary()
    print(f"\n  Wall: {wall:.2f}s  Throughput: {s.get('throughput_rps', 0):.2f} req/s  "
          f"OK/Fail: {s.get('successful', 0)}/{s.get('failed', 0)}")
    if isinstance(s.get("latency"), dict):
        print(f"  Latency P50={s['latency']['p50']:.3f}s  P95={s['latency']['p95']:.3f}s")
    return result


def run_stress_test(server_url, duration=30, concurrency=8, text_key="medium"):
    text = TEST_TEXTS[text_key]
    print(f"\n{'='*70}")
    print(f"  Stress Test ({duration}s, concurrency={concurrency})")
    print(f"{'='*70}")

    all_m = []
    stop = threading.Event()
    lock = threading.Lock()
    count = [0]

    def worker():
        while not stop.is_set():
            m = make_streaming_request(server_url, text)
            with lock:
                count[0] += 1
                all_m.append(m)
                n = count[0]
            print(f"  [{n:4d}] Lat={m.total_latency:.3f}s  Audio={m.audio_duration:.2f}s", end="\r")

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join(timeout=130)
    wall = time.perf_counter() - t0
    print()

    result = TestResult(
        test_name=f"stress_{duration}s", concurrency=concurrency,
        num_requests=len(all_m), metrics=all_m, wall_time=wall,
    )
    s = result.summary()
    print(f"  Total: {len(all_m)} requests in {wall:.1f}s = {s.get('throughput_rps', 0):.2f} req/s")
    return result


# ============================================================================
# Report
# ============================================================================

def print_report(all_results):
    print(f"\n{'='*70}")
    print(f"  FINAL REPORT")
    print(f"{'='*70}")

    hdr = f"  {'Test':<30} {'Conc':>4} {'OK':>4} {'Thru':>7} {'P50':>7} {'RTF50':>7} {'TTFA50':>8}"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for r in all_results:
        s = r.summary()
        if "error" in s:
            print(f"  {r.test_name:<30} ERROR")
            continue
        lat = s.get("latency", {})
        rtf = s.get("rtf", {})
        ttfa = s.get("ttfa", {})
        p50 = lat.get("p50", 0) if isinstance(lat, dict) else 0
        rtf50 = rtf.get("p50", 0) if isinstance(rtf, dict) else 0
        ttfa50 = ttfa.get("p50", 0) if isinstance(ttfa, dict) else 0
        print(
            f"  {s['test_name']:<30} {s['concurrency']:4d} {s['successful']:4d} "
            f"{s['throughput_rps']:6.2f}r {p50:6.3f}s {rtf50:5.2f}x {ttfa50*1000:6.0f}ms"
        )

    # Save JSON report
    report = [r.summary() for r in all_results]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Concurrency Test")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--stress-duration", type=int, default=30)
    parser.add_argument("--text", choices=["short", "medium", "long"], default="medium")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not test_health(args.server):
        print("\n  Server not ready. Start with: python server.py")
        sys.exit(1)

    all_results = []

    # Phase 1: Baseline
    print("\n" + "=" * 70)
    print("  PHASE 1: Single Request Baseline")
    print("=" * 70)
    for tk in (["medium"] if args.quick else ["short", "medium", "long"]):
        all_results.append(run_single_test(args.server, tk))

    # Phase 2: Streaming vs blocking
    print("\n" + "=" * 70)
    print("  PHASE 2: Streaming vs Non-Streaming")
    print("=" * 70)
    all_results.append(run_concurrent_test(args.server, 1, 3, args.text, streaming=True))
    all_results.append(run_concurrent_test(args.server, 1, 3, args.text, streaming=False))

    # Phase 3: Concurrency
    print("\n" + "=" * 70)
    print("  PHASE 3: Concurrency Scaling")
    print("=" * 70)
    levels = [2, 4, 8]
    if not args.quick and args.max_concurrency > 8:
        levels.append(args.max_concurrency)
    for c in levels:
        n = c if args.quick else max(c, 4)
        all_results.append(run_concurrent_test(args.server, c, n, args.text))

    # Phase 4: Stress
    if not args.quick:
        print("\n" + "=" * 70)
        print("  PHASE 4: Stress Test")
        print("=" * 70)
        all_results.append(
            run_stress_test(args.server, args.stress_duration, 8, args.text)
        )

    print_report(all_results)


if __name__ == "__main__":
    main()
