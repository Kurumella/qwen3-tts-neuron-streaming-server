#!/usr/bin/env python3
"""
Subprocess script to trace a TP Talker transformer to Neuron.

Usage:
    python trace_tp_talker.py <safetensors_path> <trace_dir> <bucket_size> <tp_degree>
"""

import sys
import os
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message="torch_neuronx.nki_jit is deprecated",
    category=DeprecationWarning,
)

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuronx_distributed.trace import parallel_model_trace, parallel_model_save
from tp_talker import TPTalkerFactory, precompute_rope


def main():
    safetensors_path = sys.argv[1]
    trace_dir = sys.argv[2]
    bucket_size = int(sys.argv[3])
    tp_degree = int(sys.argv[4])

    save_dir = os.path.join(trace_dir, f"talker_tp{tp_degree}_{bucket_size}")
    os.makedirs(save_dir, exist_ok=True)

    factory = TPTalkerFactory(safetensors_path, tp_degree=tp_degree)

    # Example inputs: (inputs_embeds, cos, sin)
    cos, sin = precompute_rope(bucket_size)
    example_inputs = (
        torch.randn(1, bucket_size, 2048),
        cos,
        sin,
    )

    print(f"[TraceTPTalker] Tracing tp={tp_degree}, bucket={bucket_size}...", flush=True)
    t0 = time.perf_counter()
    traced = parallel_model_trace(
        factory,
        example_inputs,
        tp_degree=tp_degree,
        compiler_args=(
            "--model-type=transformer "
            "--auto-cast=matmult --auto-cast-type=bf16 "
            "--tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2'"
        ),
    )
    elapsed = time.perf_counter() - t0
    print(f"[TraceTPTalker] Traced bucket {bucket_size} in {elapsed:.1f}s", flush=True)

    parallel_model_save(traced, save_dir)
    print(f"[TraceTPTalker] Saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
