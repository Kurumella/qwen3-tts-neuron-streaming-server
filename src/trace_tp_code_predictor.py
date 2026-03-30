#!/usr/bin/env python3
"""
Subprocess script to trace a TP Code Predictor to Neuron.

Usage:
    python trace_tp_code_predictor.py <safetensors_path> <trace_dir> <tp_degree>
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
from tp_code_predictor import TPCPFactory, precompute_cp_rope


def main():
    safetensors_path = sys.argv[1]
    trace_dir = sys.argv[2]
    tp_degree = int(sys.argv[3])

    save_dir = os.path.join(trace_dir, f"code_predictor_tp{tp_degree}")
    os.makedirs(save_dir, exist_ok=True)

    factory = TPCPFactory(safetensors_path, tp_degree=tp_degree)

    # Fixed seq_len=16 for code predictor
    cos, sin = precompute_cp_rope(16)
    example_inputs = (
        torch.randn(1, 16, 1024),
        cos,
        sin,
    )

    print(f"[TraceTPCP] Tracing tp={tp_degree}, seq_len=16...", flush=True)
    t0 = time.perf_counter()
    traced = parallel_model_trace(
        factory,
        example_inputs,
        tp_degree=tp_degree,
        compiler_args=(
            "--model-type=transformer "
            "--auto-cast=matmult --auto-cast-type=bf16"
        ),
    )
    elapsed = time.perf_counter() - t0
    print(f"[TraceTPCP] Traced in {elapsed:.1f}s", flush=True)

    parallel_model_save(traced, save_dir)
    print(f"[TraceTPCP] Saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
