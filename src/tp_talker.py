"""
tp_talker.py — Tensor-Parallel Talker Transformer for Qwen3-TTS (TP=2)

Shards the 28-layer Talker transformer across 2 NeuronCores using
neuronx_distributed ColumnParallelLinear and RowParallelLinear.

Sharding strategy per transformer block:
  - q_proj, k_proj, v_proj:  ColumnParallel — each core gets half the heads
  - o_proj:                   RowParallel    — all-reduced across cores
  - gate_proj, up_proj:       ColumnParallel — each core gets half intermediate dim
  - down_proj:                RowParallel    — all-reduced across cores
  - RMSNorm, q_norm, k_norm: Replicated     — identical on both cores

Architecture:
  - hidden_size: 2048
  - num_layers: 28
  - num_attention_heads: 16 (head_dim=128)
  - num_key_value_heads: 8 (GQA, 2 groups)
  - intermediate_size: 6144
  - hidden_act: silu
  - rms_norm_eps: 1e-6
  - rope_theta: 1_000_000
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


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TPTalkerBlock(nn.Module):
    """Single Talker transformer block with tensor-parallel attention and MLP.

    GQA: 16 query heads, 8 KV heads. With TP=2, each core gets 8 Q heads and 4 KV heads.
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size,
                 head_dim, eps, tp_degree):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tp_degree = tp_degree

        # Local (per-core) head counts
        self.local_q_heads = num_heads // tp_degree
        self.local_kv_heads = num_kv_heads // tp_degree
        self.num_kv_groups = self.local_q_heads // self.local_kv_heads

        # Pre-attention norm (replicated)
        self.input_layernorm = RMSNorm(hidden_size, eps)

        # Attention projections — sharded across cores
        # Q: ColumnParallel — each core gets local_q_heads * head_dim outputs
        self.q_proj = ColumnParallelLinear(
            hidden_size, num_heads * head_dim,
            bias=False, gather_output=False,
        )
        # K: ColumnParallel — each core gets local_kv_heads * head_dim outputs
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim,
            bias=False, gather_output=False,
        )
        # V: ColumnParallel
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim,
            bias=False, gather_output=False,
        )
        # O: RowParallel — all-reduce across cores
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size,
            bias=False, input_is_parallel=True,
        )

        # Per-head RMSNorm (on local head dim, replicated)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

        # Post-attention norm (replicated)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)

        # SiLU gated MLP — sharded
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size,
            bias=False, gather_output=False,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size,
            bias=False, gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            bias=False, input_is_parallel=True,
        )

    def forward(self, hidden_states, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        batch, seq_len, _ = hidden_states.shape

        # QKV projections — outputs are already sharded by ColumnParallel
        q = self.q_proj(hidden_states)  # (B, S, local_q_heads * head_dim)
        k = self.k_proj(hidden_states)  # (B, S, local_kv_heads * head_dim)
        v = self.v_proj(hidden_states)  # (B, S, local_kv_heads * head_dim)

        q = q.view(batch, seq_len, self.local_q_heads, self.head_dim)
        k = k.view(batch, seq_len, self.local_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.local_kv_heads, self.head_dim)

        # Per-head RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (B, heads, S, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA expansion (within local heads)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Causal attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch, seq_len, self.local_q_heads * self.head_dim
        )
        attn_out = self.o_proj(attn_out)  # RowParallel: all-reduce
        hidden_states = residual + attn_out

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * up)  # RowParallel: all-reduce
        hidden_states = residual + hidden_states

        return hidden_states


class TPTalkerTransformer(nn.Module):
    """Full Talker transformer with tensor parallelism.

    Forward: (inputs_embeds, cos, sin) -> hidden_states
    """

    def __init__(self, num_layers=28, hidden_size=2048, num_heads=16,
                 num_kv_heads=8, intermediate_size=6144, head_dim=128,
                 eps=1e-6, tp_degree=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TPTalkerBlock(hidden_size, num_heads, num_kv_heads,
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


class TPTalkerFactory:
    """Picklable factory for parallel_model_trace.

    Creates a TP-sharded Talker and loads rank-specific weights from safetensors.
    """

    def __init__(self, safetensors_path, tp_degree=2):
        self.safetensors_path = safetensors_path
        self.tp_degree = tp_degree

    def __call__(self):
        model = TPTalkerTransformer(tp_degree=self.tp_degree)
        model.eval()

        rank = parallel_state.get_tensor_model_parallel_rank()
        tp = self.tp_degree
        self._load_sharded_weights(model, rank, tp)

        return model, None

    def _load_sharded_weights(self, model, rank, tp):
        from safetensors import safe_open

        with safe_open(self.safetensors_path, framework="pt") as f:
            for i, layer in enumerate(model.layers):
                prefix = f"talker.model.layers.{i}"
                head_dim = 128

                # --- Q projection: ColumnParallel ---
                # Full shape: (num_heads * head_dim, hidden) = (2048, 2048)
                # Shard output dim: each rank gets local_q_heads * head_dim rows
                w = f.get_tensor(f"{prefix}.self_attn.q_proj.weight")  # (2048, 2048)
                num_heads = w.shape[0] // head_dim  # 16
                local_heads = num_heads // tp  # 8
                chunk = local_heads * head_dim  # 1024
                layer.q_proj.weight.data.copy_(w[rank * chunk:(rank + 1) * chunk])

                # --- K projection: ColumnParallel ---
                w = f.get_tensor(f"{prefix}.self_attn.k_proj.weight")  # (1024, 2048)
                num_kv_heads = w.shape[0] // head_dim  # 8
                local_kv = num_kv_heads // tp  # 4
                kv_chunk = local_kv * head_dim  # 512
                layer.k_proj.weight.data.copy_(w[rank * kv_chunk:(rank + 1) * kv_chunk])

                # --- V projection: ColumnParallel ---
                w = f.get_tensor(f"{prefix}.self_attn.v_proj.weight")  # (1024, 2048)
                layer.v_proj.weight.data.copy_(w[rank * kv_chunk:(rank + 1) * kv_chunk])

                # --- O projection: RowParallel ---
                # Full shape: (hidden, num_heads * head_dim) = (2048, 2048)
                # Shard input dim (columns)
                w = f.get_tensor(f"{prefix}.self_attn.o_proj.weight")  # (2048, 2048)
                col_chunk = chunk  # 1024
                layer.o_proj.weight.data.copy_(w[:, rank * col_chunk:(rank + 1) * col_chunk])

                # --- Per-head norms: replicated (operate on head_dim=128) ---
                layer.q_norm.weight.data.copy_(f.get_tensor(f"{prefix}.self_attn.q_norm.weight"))
                layer.k_norm.weight.data.copy_(f.get_tensor(f"{prefix}.self_attn.k_norm.weight"))

                # --- Layer norms: replicated ---
                layer.input_layernorm.weight.data.copy_(
                    f.get_tensor(f"{prefix}.input_layernorm.weight"))
                layer.post_attention_layernorm.weight.data.copy_(
                    f.get_tensor(f"{prefix}.post_attention_layernorm.weight"))

                # --- gate_proj: ColumnParallel ---
                # Full shape: (intermediate, hidden) = (6144, 2048)
                w = f.get_tensor(f"{prefix}.mlp.gate_proj.weight")
                mlp_chunk = w.shape[0] // tp  # 3072
                layer.gate_proj.weight.data.copy_(w[rank * mlp_chunk:(rank + 1) * mlp_chunk])

                # --- up_proj: ColumnParallel ---
                w = f.get_tensor(f"{prefix}.mlp.up_proj.weight")
                layer.up_proj.weight.data.copy_(w[rank * mlp_chunk:(rank + 1) * mlp_chunk])

                # --- down_proj: RowParallel ---
                # Full shape: (hidden, intermediate) = (2048, 6144)
                w = f.get_tensor(f"{prefix}.mlp.down_proj.weight")
                layer.down_proj.weight.data.copy_(w[:, rank * mlp_chunk:(rank + 1) * mlp_chunk])

            # Final norm: replicated
            model.norm.weight.data.copy_(f.get_tensor("talker.model.norm.weight"))

        print(f"[TPTalker] Rank {rank}: loaded sharded weights")


def precompute_rope(max_seq_len, head_dim=128, theta=1_000_000.0):
    """Pre-compute cos/sin for RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin
