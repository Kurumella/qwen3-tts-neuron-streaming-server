"""
tp_code_predictor.py — Tensor-Parallel Code Predictor for Qwen3-TTS (TP=2)

Shards the 5-layer Code Predictor transformer across 2 NeuronCores.

Architecture:
  - hidden_size: 1024
  - num_layers: 5
  - num_attention_heads: 16 (head_dim=128)
  - num_key_value_heads: 8 (GQA)
  - intermediate_size: 3072
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="torch_neuronx.nki_jit is deprecated",
    category=DeprecationWarning,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers import parallel_state


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class TPCPBlock(nn.Module):
    """Code Predictor block with tensor parallelism."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size,
                 head_dim, eps, tp_degree):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tp_degree = tp_degree

        self.local_q_heads = num_heads // tp_degree
        self.local_kv_heads = num_kv_heads // tp_degree
        self.num_kv_groups = self.local_q_heads // self.local_kv_heads

        self.input_layernorm = RMSNorm(hidden_size, eps)

        self.q_proj = ColumnParallelLinear(
            hidden_size, num_heads * head_dim, bias=False, gather_output=False)
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False, gather_output=False)
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False, gather_output=False)
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size, bias=False, input_is_parallel=True)

        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

        self.post_attention_layernorm = RMSNorm(hidden_size, eps)

        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False)
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True)

    def forward(self, hidden_states, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch, seq_len, self.local_q_heads, self.head_dim)
        k = k.view(batch, seq_len, self.local_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.local_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        if self.num_kv_groups > 1:
            k_embed = k_embed.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(q_embed, k_embed, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch, seq_len, self.local_q_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states


class TPCPTransformer(nn.Module):
    """TP Code Predictor transformer (5 layers + final norm)."""

    def __init__(self, num_layers=5, hidden_size=1024, num_heads=16,
                 num_kv_heads=8, intermediate_size=3072, head_dim=128,
                 eps=1e-6, tp_degree=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TPCPBlock(hidden_size, num_heads, num_kv_heads,
                      intermediate_size, head_dim, eps, tp_degree)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps)

    def forward(self, inputs_embeds, cos, sin):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class TPCPFactory:
    """Picklable factory for parallel_model_trace."""

    def __init__(self, safetensors_path, tp_degree=2):
        self.safetensors_path = safetensors_path
        self.tp_degree = tp_degree

    def __call__(self):
        model = TPCPTransformer(tp_degree=self.tp_degree)
        model.eval()

        rank = parallel_state.get_tensor_model_parallel_rank()
        tp = self.tp_degree
        self._load_sharded_weights(model, rank, tp)

        return model, None

    def _load_sharded_weights(self, model, rank, tp):
        from safetensors import safe_open

        head_dim = 128

        with safe_open(self.safetensors_path, framework="pt") as f:
            for i, layer in enumerate(model.layers):
                prefix = f"talker.code_predictor.model.layers.{i}"

                # Q: ColumnParallel (2048, 1024) -> shard output
                w = f.get_tensor(f"{prefix}.self_attn.q_proj.weight")  # (2048, 1024)
                num_heads = w.shape[0] // head_dim  # 16
                local_heads = num_heads // tp  # 8
                chunk = local_heads * head_dim  # 1024
                layer.q_proj.weight.data.copy_(w[rank * chunk:(rank + 1) * chunk])

                # K: ColumnParallel (1024, 1024) -> shard output
                w = f.get_tensor(f"{prefix}.self_attn.k_proj.weight")  # (1024, 1024)
                num_kv = w.shape[0] // head_dim  # 8
                local_kv = num_kv // tp  # 4
                kv_chunk = local_kv * head_dim  # 512
                layer.k_proj.weight.data.copy_(w[rank * kv_chunk:(rank + 1) * kv_chunk])

                # V: ColumnParallel
                w = f.get_tensor(f"{prefix}.self_attn.v_proj.weight")
                layer.v_proj.weight.data.copy_(w[rank * kv_chunk:(rank + 1) * kv_chunk])

                # O: RowParallel (1024, 2048) -> shard input (columns)
                w = f.get_tensor(f"{prefix}.self_attn.o_proj.weight")  # (1024, 2048)
                layer.o_proj.weight.data.copy_(w[:, rank * chunk:(rank + 1) * chunk])

                # Per-head norms: replicated
                layer.q_norm.weight.data.copy_(f.get_tensor(f"{prefix}.self_attn.q_norm.weight"))
                layer.k_norm.weight.data.copy_(f.get_tensor(f"{prefix}.self_attn.k_norm.weight"))

                # Layer norms: replicated
                layer.input_layernorm.weight.data.copy_(
                    f.get_tensor(f"{prefix}.input_layernorm.weight"))
                layer.post_attention_layernorm.weight.data.copy_(
                    f.get_tensor(f"{prefix}.post_attention_layernorm.weight"))

                # gate_proj: ColumnParallel (3072, 1024)
                w = f.get_tensor(f"{prefix}.mlp.gate_proj.weight")
                mlp_chunk = w.shape[0] // tp  # 1536
                layer.gate_proj.weight.data.copy_(w[rank * mlp_chunk:(rank + 1) * mlp_chunk])

                # up_proj: ColumnParallel
                w = f.get_tensor(f"{prefix}.mlp.up_proj.weight")
                layer.up_proj.weight.data.copy_(w[rank * mlp_chunk:(rank + 1) * mlp_chunk])

                # down_proj: RowParallel (1024, 3072)
                w = f.get_tensor(f"{prefix}.mlp.down_proj.weight")
                layer.down_proj.weight.data.copy_(w[:, rank * mlp_chunk:(rank + 1) * mlp_chunk])

            # Final norm: replicated
            model.norm.weight.data.copy_(
                f.get_tensor("talker.code_predictor.model.norm.weight"))

        print(f"[TPCP] Rank {rank}: loaded sharded weights")


def precompute_cp_rope(max_seq_len=16, head_dim=128, theta=1_000_000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin
