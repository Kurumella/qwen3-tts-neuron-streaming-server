"""
QwenTTS Neuron Pipeline for 12Hz 1.7B CustomVoice model.

Runs the Talker transformer on NeuronCores via torch_neuronx.trace.
Code predictor, embeddings, and speech decoder run on CPU.

Usage:
    pipeline = QwenTTSNeuronPipeline(
        model_dir="~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/...",
        trace_dir="/tmp/qwen_tts_neuron_traces",
    )
    wavs, sr = pipeline.generate("Hello world!", speaker="Ryan", language="English")
"""

import json
import os
import subprocess
import sys
import time
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings(
    "ignore",
    message="torch_neuronx.nki_jit is deprecated",
    category=DeprecationWarning,
)

from neuronx_distributed.trace import parallel_model_load

from tp_talker import precompute_rope
from tp_code_predictor import precompute_cp_rope


# --------------------------------------------------------------------------- #
#  Config / Constants
# --------------------------------------------------------------------------- #

# 1.7B CustomVoice model architecture
HIDDEN_SIZE = 2048
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
NUM_LAYERS = 28
INTERMEDIATE_SIZE = 6144
ROPE_THETA = 1_000_000.0
RMS_NORM_EPS = 1e-6
VOCAB_SIZE = 3072  # codec vocab
TEXT_VOCAB_SIZE = 151936
TEXT_HIDDEN_SIZE = 2048

# Code predictor
CP_HIDDEN_SIZE = 1024
CP_INTERMEDIATE_SIZE = 3072
CP_NUM_LAYERS = 5
CP_NUM_HEADS = 16
CP_NUM_KV_HEADS = 8
CP_VOCAB_SIZE = 2048
CP_NUM_CODE_GROUPS = 16
CP_HEAD_DIM = 128

# Bucket sizes for Neuron tracing
BUCKET_SIZES = [64, 128, 256, 512, 1024]
MAX_SEQ_LEN = max(BUCKET_SIZES)


# --------------------------------------------------------------------------- #
#  CPU-based Code Predictor
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CPBlock(nn.Module):
    """Code Predictor transformer block (CPU) with KV-cache support."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size, head_dim, eps):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states, cos, sin, k_cache=None, v_cache=None):
        """
        Forward with optional KV-cache.
        If k_cache/v_cache are provided, hidden_states is the new token only (seq_len=1).
        Returns: (output, new_k_cache, new_v_cache)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # Update KV cache
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_k_cache = k
        new_v_cache = v

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            k_expanded = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_expanded = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_expanded = k
            v_expanded = v

        attn_out = F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=(k_cache is None))
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_out = self.o_proj(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, new_k_cache, new_v_cache


class CPUCodePredictor(nn.Module):
    """
    Code Predictor: predicts codebooks 1-15 given the talker's last hidden state
    and the first codebook token. Runs entirely on CPU.
    """

    def __init__(self):
        super().__init__()
        # Projection from talker hidden (2048) to CP hidden (1024)
        self.small_to_mtp_projection = nn.Linear(HIDDEN_SIZE, CP_HIDDEN_SIZE, bias=True)

        # Transformer layers
        self.layers = nn.ModuleList([
            CPBlock(CP_HIDDEN_SIZE, CP_NUM_HEADS, CP_NUM_KV_HEADS,
                    CP_INTERMEDIATE_SIZE, CP_HEAD_DIM, RMS_NORM_EPS)
            for _ in range(CP_NUM_LAYERS)
        ])
        self.norm = RMSNorm(CP_HIDDEN_SIZE, RMS_NORM_EPS)

        # 15 codec embeddings (vocab=2048, dim=2048 to match talker space)
        self.codec_embeddings = nn.ModuleList([
            nn.Embedding(CP_VOCAB_SIZE, HIDDEN_SIZE) for _ in range(CP_NUM_CODE_GROUPS - 1)
        ])

        # 15 LM heads (from CP hidden 1024 to vocab 2048)
        self.lm_heads = nn.ModuleList([
            nn.Linear(CP_HIDDEN_SIZE, CP_VOCAB_SIZE, bias=False) for _ in range(CP_NUM_CODE_GROUPS - 1)
        ])

        # RoPE for code predictor
        self._cp_cos = None
        self._cp_sin = None

    def _get_cp_rope(self, seq_len, device):
        if self._cp_cos is None or self._cp_cos.shape[2] < seq_len:
            max_len = max(seq_len, 64)
            inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, CP_HEAD_DIM, 2, dtype=torch.float32) / CP_HEAD_DIM))
            positions = torch.arange(max_len, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cp_cos = emb.cos().unsqueeze(0).unsqueeze(0)
            self._cp_sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return self._cp_cos[:, :, :seq_len].to(device), self._cp_sin[:, :, :seq_len].to(device)

    @torch.inference_mode()
    def generate(self, talker_hidden, first_code_token, first_code_embed,
                 do_sample=True, top_k=50, top_p=1.0, temperature=0.9):
        """
        Generate codebooks 1-15 autoregressively with KV-caching.

        Args:
            talker_hidden: (1, 1, 2048) - last hidden state from talker
            first_code_token: int - the first codebook token (from codec_head sampling)
            first_code_embed: (1, 2048) - embedding of first_code_token from talker's codec_embedding
            do_sample: bool
            top_k, top_p, temperature: sampling params

        Returns:
            all_codes: list of 16 ints (codebook 0 + 15 predicted)
            all_code_embeddings_sum: (1, 2048) - sum of all 16 codebook embeddings (in talker space)
        """
        device = talker_hidden.device
        all_codes = [first_code_token]

        # Prefill: [talker_hidden, first_code_embed] → 2 positions
        # Position 0: projected talker hidden
        # Position 1: projected first code embedding (from talker's codec_embedding)
        prefill_embeds = torch.cat([
            talker_hidden,  # (1, 1, 2048)
            first_code_embed.unsqueeze(0) if first_code_embed.dim() == 2 else first_code_embed,  # (1, 1, 2048)
        ], dim=1)  # (1, 2, 2048)
        projected = self.small_to_mtp_projection(prefill_embeds)  # (1, 2, 1024)

        kv_caches = [None] * CP_NUM_LAYERS
        cos, sin = self._get_cp_rope(2, device)  # positions 0 and 1

        hidden = projected
        for li, layer in enumerate(self.layers):
            hidden, k_cache, v_cache = layer(hidden, cos, sin)
            kv_caches[li] = (k_cache, v_cache)
        hidden = self.norm(hidden)

        # Get first prediction from position 1 (the first_code_embed position)
        logits = self.lm_heads[0](hidden[:, -1:, :]).squeeze(0).squeeze(0)
        token = self._sample(logits, do_sample, top_k, top_p, temperature)
        all_codes.append(token)

        # Generate remaining 13 codebook tokens with KV-cache (positions 2-15)
        for step in range(1, CP_NUM_CODE_GROUPS - 1):
            # Embed previous token using CP's codec_embedding and project
            token_embed = self.codec_embeddings[step - 1](
                torch.tensor([[token]], dtype=torch.long, device=device)
            )  # (1, 1, 2048)
            token_projected = self.small_to_mtp_projection(token_embed)  # (1, 1, 1024)

            # RoPE for current position (shifted by 1 for the 2-token prefill)
            pos = step + 1
            cos, sin = self._get_cp_rope(pos + 1, device)
            cos = cos[:, :, pos:pos + 1, :]
            sin = sin[:, :, pos:pos + 1, :]

            hidden = token_projected
            for li, layer in enumerate(self.layers):
                k_cache, v_cache = kv_caches[li]
                hidden, k_cache, v_cache = layer(hidden, cos, sin, k_cache, v_cache)
                kv_caches[li] = (k_cache, v_cache)
            hidden = self.norm(hidden)

            logits = self.lm_heads[step](hidden[:, -1:, :]).squeeze(0).squeeze(0)
            token = self._sample(logits, do_sample, top_k, top_p, temperature)
            all_codes.append(token)

        # Compute sum of all codebook embeddings in talker (2048) space
        cp_embeds_sum = torch.zeros(1, HIDDEN_SIZE, device=device)
        for i in range(CP_NUM_CODE_GROUPS - 1):
            cp_embeds_sum += self.codec_embeddings[i](
                torch.tensor([all_codes[i + 1]], dtype=torch.long, device=device)
            ).squeeze(0)

        return all_codes, cp_embeds_sum

    @staticmethod
    def _sample(logits, do_sample, top_k, top_p, temperature):
        if not do_sample:
            return logits.argmax().item()

        logits = logits / max(temperature, 1e-6)

        # Top-k
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()


def load_code_predictor(safetensors_path: str) -> CPUCodePredictor:
    """Load the code predictor weights from safetensors."""
    from safetensors import safe_open

    cp = CPUCodePredictor()

    with safe_open(safetensors_path, framework="pt") as f:
        # Projection
        cp.small_to_mtp_projection.weight.data.copy_(
            f.get_tensor("talker.code_predictor.small_to_mtp_projection.weight"))
        cp.small_to_mtp_projection.bias.data.copy_(
            f.get_tensor("talker.code_predictor.small_to_mtp_projection.bias"))

        # Transformer layers
        for i in range(CP_NUM_LAYERS):
            prefix = f"talker.code_predictor.model.layers.{i}"
            layer = cp.layers[i]
            layer.q_proj.weight.data.copy_(f.get_tensor(f"{prefix}.self_attn.q_proj.weight"))
            layer.k_proj.weight.data.copy_(f.get_tensor(f"{prefix}.self_attn.k_proj.weight"))
            layer.v_proj.weight.data.copy_(f.get_tensor(f"{prefix}.self_attn.v_proj.weight"))
            layer.o_proj.weight.data.copy_(f.get_tensor(f"{prefix}.self_attn.o_proj.weight"))
            layer.q_norm.weight.data.copy_(f.get_tensor(f"{prefix}.self_attn.q_norm.weight"))
            layer.k_norm.weight.data.copy_(f.get_tensor(f"{prefix}.self_attn.k_norm.weight"))
            layer.input_layernorm.weight.data.copy_(f.get_tensor(f"{prefix}.input_layernorm.weight"))
            layer.post_attention_layernorm.weight.data.copy_(f.get_tensor(f"{prefix}.post_attention_layernorm.weight"))
            layer.gate_proj.weight.data.copy_(f.get_tensor(f"{prefix}.mlp.gate_proj.weight"))
            layer.up_proj.weight.data.copy_(f.get_tensor(f"{prefix}.mlp.up_proj.weight"))
            layer.down_proj.weight.data.copy_(f.get_tensor(f"{prefix}.mlp.down_proj.weight"))

        # Final norm
        cp.norm.weight.data.copy_(f.get_tensor("talker.code_predictor.model.norm.weight"))

        # Codec embeddings (15)
        for i in range(CP_NUM_CODE_GROUPS - 1):
            cp.codec_embeddings[i].weight.data.copy_(
                f.get_tensor(f"talker.code_predictor.model.codec_embedding.{i}.weight"))

        # LM heads (15)
        for i in range(CP_NUM_CODE_GROUPS - 1):
            cp.lm_heads[i].weight.data.copy_(
                f.get_tensor(f"talker.code_predictor.lm_head.{i}.weight"))

    cp.eval()
    print(f"[CodePredictor] Loaded weights from {safetensors_path}")
    return cp


# --------------------------------------------------------------------------- #
#  Neuron-accelerated Code Predictor
# --------------------------------------------------------------------------- #

class NeuronCodePredictor:
    """
    Code Predictor using Neuron-traced transformer.

    The transformer is traced for seq_len=16 (fixed). The autoregressive loop
    runs on host, calling the Neuron model each step with padded input.
    Since causal attention is used, the padded positions don't affect earlier ones.
    """

    def __init__(self, neuron_model, safetensors_path: str):
        self.neuron_model = neuron_model

        # Pre-compute RoPE for CP
        self.cp_cos, self.cp_sin = precompute_cp_rope(16, CP_HEAD_DIM)

        # Load non-transformer weights on CPU
        from safetensors import safe_open

        self.small_to_mtp_projection_w = None
        self.small_to_mtp_projection_b = None
        self.codec_embeddings = []  # List of weight tensors
        self.lm_heads = []  # List of weight tensors

        with safe_open(safetensors_path, framework="pt") as f:
            self.small_to_mtp_projection_w = f.get_tensor(
                "talker.code_predictor.small_to_mtp_projection.weight").clone().float()
            self.small_to_mtp_projection_b = f.get_tensor(
                "talker.code_predictor.small_to_mtp_projection.bias").clone().float()

            for i in range(CP_NUM_CODE_GROUPS - 1):
                self.codec_embeddings.append(
                    f.get_tensor(f"talker.code_predictor.model.codec_embedding.{i}.weight").clone().float())
                self.lm_heads.append(
                    f.get_tensor(f"talker.code_predictor.lm_head.{i}.weight").clone().float())

        print(f"[NeuronCP] Loaded non-transformer weights")

    def _project(self, x):
        """Project from talker space (2048) to CP space (1024)."""
        return F.linear(x, self.small_to_mtp_projection_w, self.small_to_mtp_projection_b)

    def _embed_code(self, code_idx, code_token):
        """Embed a code token using the code_idx-th embedding table."""
        return self.codec_embeddings[code_idx][code_token]  # (2048,)

    def _get_logits(self, hidden, head_idx):
        """Get logits from the head_idx-th LM head."""
        return F.linear(hidden, self.lm_heads[head_idx])  # (2048,)

    @torch.inference_mode()
    def generate(self, talker_hidden, first_code_token, first_code_embed,
                 do_sample=True, top_k=50, top_p=1.0, temperature=0.9):
        """
        Generate codebooks 1-15 using Neuron-traced transformer.

        The approach: build a 16-position buffer, fill positions incrementally,
        and call the Neuron CP transformer each step. Due to causal attention,
        padding doesn't affect earlier positions.

        Buffer layout (matches reference):
          Position 0: projected talker hidden state
          Position 1: projected first code embedding (codebook 0, from talker codec_embedding)
          Position 2: projected CP codec_embedding[0](codebook 1 token)
          ...
          Position 15: projected CP codec_embedding[13](codebook 14 token)
        """
        all_codes = [first_code_token]

        # Pre-allocate 16-position buffer
        buffer = torch.zeros(1, 16, CP_HIDDEN_SIZE)

        # Position 0: projected talker hidden
        buffer[0, 0, :] = self._project(talker_hidden.squeeze(0).squeeze(0))

        # Position 1: projected first code embedding (from talker's codec_embedding)
        buffer[0, 1, :] = self._project(first_code_embed.squeeze())

        # Run transformer (prefill) and get first prediction at position 1
        hidden_out = self.neuron_model(buffer, self.cp_cos, self.cp_sin)

        logits = self._get_logits(hidden_out[0, 1, :], 0)  # lm_head[0] at position 1
        token = _sample_token_fast(logits, do_sample, top_k, top_p, temperature)
        all_codes.append(token)

        # Generate remaining 13 codes (codebooks 2-14) at positions 2-14
        for step in range(1, CP_NUM_CODE_GROUPS - 1):
            # Embed previous code using CP's codec_embedding and project to CP space
            code_embed = self._embed_code(step - 1, token)  # (2048,)
            code_projected = self._project(code_embed)  # (1024,)
            buffer[0, step + 1, :] = code_projected  # Position step+1 (shifted by 1 for prefill)

            # Run full transformer (causal attention handles padding)
            hidden_out = self.neuron_model(buffer, self.cp_cos, self.cp_sin)

            logits = self._get_logits(hidden_out[0, step + 1, :], step)  # lm_head[step] at position step+1
            token = _sample_token_fast(logits, do_sample, top_k, top_p, temperature)
            all_codes.append(token)

        # Compute sum of all codebook embeddings in talker (2048) space
        cp_embeds_sum = torch.zeros(1, HIDDEN_SIZE)
        for i in range(CP_NUM_CODE_GROUPS - 1):
            cp_embeds_sum += self._embed_code(i, all_codes[i + 1]).unsqueeze(0)

        return all_codes, cp_embeds_sum


def _sample_token_fast(logits, do_sample, top_k, top_p, temperature):
    """Fast token sampling without class overhead."""
    if not do_sample:
        return logits.argmax().item()

    logits = logits.float() / max(temperature, 1e-6)

    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
        logits[indices_to_remove] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


# --------------------------------------------------------------------------- #
#  CPU Embeddings and Projections
# --------------------------------------------------------------------------- #

class CPUEmbeddings(nn.Module):
    """All embedding and projection layers that run on CPU."""

    def __init__(self):
        super().__init__()
        # Codec embedding (codebook 0 in talker space)
        self.codec_embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        # Text embedding
        self.text_embedding = nn.Embedding(TEXT_VOCAB_SIZE, TEXT_HIDDEN_SIZE)
        # Text projection MLP: 2048 -> SiLU -> 2048 -> 2048
        self.text_proj_fc1 = nn.Linear(TEXT_HIDDEN_SIZE, TEXT_HIDDEN_SIZE, bias=True)
        self.text_proj_fc2 = nn.Linear(TEXT_HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
        # Codec head (logits for codebook 0)
        self.codec_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def text_projection(self, text_embeds):
        return self.text_proj_fc2(F.silu(self.text_proj_fc1(text_embeds)))

    def get_codec_logits(self, hidden_states):
        return self.codec_head(hidden_states)


def load_embeddings(safetensors_path: str) -> CPUEmbeddings:
    from safetensors import safe_open

    emb = CPUEmbeddings()

    with safe_open(safetensors_path, framework="pt") as f:
        emb.codec_embedding.weight.data.copy_(f.get_tensor("talker.model.codec_embedding.weight"))
        emb.text_embedding.weight.data.copy_(f.get_tensor("talker.model.text_embedding.weight"))
        emb.text_proj_fc1.weight.data.copy_(f.get_tensor("talker.text_projection.linear_fc1.weight"))
        emb.text_proj_fc1.bias.data.copy_(f.get_tensor("talker.text_projection.linear_fc1.bias"))
        emb.text_proj_fc2.weight.data.copy_(f.get_tensor("talker.text_projection.linear_fc2.weight"))
        emb.text_proj_fc2.bias.data.copy_(f.get_tensor("talker.text_projection.linear_fc2.bias"))
        emb.codec_head.weight.data.copy_(f.get_tensor("talker.codec_head.weight"))

    emb.eval()
    print(f"[Embeddings] Loaded weights from {safetensors_path}")
    return emb


# --------------------------------------------------------------------------- #
#  Main Pipeline
# --------------------------------------------------------------------------- #

class QwenTTSNeuronPipeline:
    """
    QwenTTS 12Hz 1.7B Neuron-accelerated pipeline.

    - Talker transformer runs on NeuronCores (traced for multiple bucket sizes)
    - Code predictor, embeddings, and speech decoder run on CPU
    """

    def __init__(
        self,
        model_dir: str,
        trace_dir: str = "/tmp/qwen_tts_neuron_traces",
        bucket_sizes: list = None,
        force_trace: bool = False,
        speech_tokenizer_dir: str = None,
        tp_degree: int = 2,
    ):
        """
        Args:
            model_dir: Path to HuggingFace model directory containing model.safetensors and config.json
            trace_dir: Where to save/load traced Neuron models
            bucket_sizes: List of bucket sizes for tracing
            force_trace: If True, re-trace even if cached
            speech_tokenizer_dir: Path to speech tokenizer model dir (if None, extracted from model_dir)
            tp_degree: Tensor parallelism degree (default 2 for trn1.2xlarge with 2 NeuronCores)
        """
        self.model_dir = model_dir
        self.trace_dir = trace_dir
        self.bucket_sizes = sorted(bucket_sizes or BUCKET_SIZES)
        self.max_seq_len = max(self.bucket_sizes)
        self.force_trace = force_trace
        self.tp_degree = tp_degree

        # Determine Python command for subprocess tracing:
        #   - If TTS_PYTHON_CMD env var is set, use it (e.g., "python3 -u")
        #   - If conda 'ldm' env exists, use "conda run -n ldm python -u"
        #   - Otherwise, use "python3 -u"
        self._python_cmd = self._resolve_python_cmd()

        safetensors_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"model.safetensors not found at {safetensors_path}")

        self.safetensors_path = safetensors_path

        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            self.config = json.load(f)
        self.talker_config = self.config["talker_config"]

        # Speech tokenizer path
        self.speech_tokenizer_dir = speech_tokenizer_dir or os.path.join(model_dir, "speech_tokenizer")

        # Load CPU components
        print("[Pipeline] Loading CPU components...")
        self.embeddings = load_embeddings(safetensors_path)

        # Pre-compute RoPE
        print("[Pipeline] Pre-computing RoPE...")
        self.rope_cos, self.rope_sin = precompute_rope(
            self.max_seq_len, head_dim=HEAD_DIM, theta=ROPE_THETA
        )

        # Trace or load Neuron models (talker + code predictor)
        self.neuron_talker_buckets = {}
        self.code_predictor = None
        self._trace_or_load_neuron_models()

        # Load speech tokenizer (for decoding codes -> audio)
        self.speech_tokenizer = None
        self._load_speech_tokenizer()

        # Load text tokenizer
        self._load_text_tokenizer()

        # Warmup
        self._warmup()

        print("[Pipeline] Ready!")

    @staticmethod
    def _resolve_python_cmd():
        """Determine the Python command for subprocess tracing."""
        env_cmd = os.environ.get("TTS_PYTHON_CMD")
        if env_cmd:
            return env_cmd.split()

        # Check if conda 'ldm' env exists
        import shutil
        if shutil.which("conda"):
            try:
                result = subprocess.run(
                    ["conda", "env", "list"],
                    capture_output=True, text=True, timeout=10,
                )
                if "ldm" in result.stdout:
                    return ["conda", "run", "-n", "ldm", "python", "-u"]
            except Exception:
                pass

        return ["python3", "-u"]

    def _trace_or_load_neuron_models(self):
        """Trace talker + code predictor using neuronx_distributed TP, or load cached."""
        os.makedirs(self.trace_dir, exist_ok=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tp = self.tp_degree

        # --- Talker buckets (TP-sharded) ---
        trace_script = os.path.join(script_dir, "trace_tp_talker.py")
        for bsize in self.bucket_sizes:
            save_dir = os.path.join(self.trace_dir, f"talker_tp{tp}_{bsize}")

            if os.path.isdir(save_dir) and not self.force_trace:
                print(f"[Pipeline] Loading cached TP talker for bucket={bsize} (tp={tp})")
                self.neuron_talker_buckets[bsize] = parallel_model_load(save_dir)
            else:
                print(f"[Pipeline] Tracing TP talker for bucket={bsize} (tp={tp}, subprocess)...")
                result = subprocess.run(
                    [
                        *self._python_cmd,
                        trace_script,
                        self.safetensors_path,
                        self.trace_dir,
                        str(bsize),
                        str(tp),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
                print(result.stdout)
                if result.returncode != 0:
                    print(f"[Pipeline] STDERR: {result.stderr}")
                    raise RuntimeError(f"TP tracing failed for bucket={bsize}: {result.stderr[-500:]}")

                self.neuron_talker_buckets[bsize] = parallel_model_load(save_dir)
                print(f"[Pipeline] Loaded TP talker for bucket={bsize}")

        # --- Code Predictor (TP-sharded) ---
        cp_trace_script = os.path.join(script_dir, "trace_tp_code_predictor.py")
        cp_save_dir = os.path.join(self.trace_dir, f"code_predictor_tp{tp}")

        if os.path.isdir(cp_save_dir) and not self.force_trace:
            print(f"[Pipeline] Loading cached TP code predictor (tp={tp})")
            cp_neuron = parallel_model_load(cp_save_dir)
        else:
            print(f"[Pipeline] Tracing TP code predictor (tp={tp}, subprocess)...")
            result = subprocess.run(
                [
                    *self._python_cmd,
                    cp_trace_script,
                    self.safetensors_path,
                    self.trace_dir,
                    str(tp),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"[Pipeline] STDERR: {result.stderr}")
                raise RuntimeError(f"TP code predictor tracing failed: {result.stderr[-500:]}")
            cp_neuron = parallel_model_load(cp_save_dir)

        self.code_predictor = NeuronCodePredictor(cp_neuron, self.safetensors_path)
        print("[Pipeline] Neuron code predictor ready")

    def _load_speech_tokenizer(self):
        """Load the 12Hz speech tokenizer for decoding codes to audio."""
        # Try to load Neuron-traced speech decoder first
        self._neuron_speech_decoder = None
        self._neuron_speech_chunk_size = None
        self._speech_decode_upsample = 1920  # Total upsample rate for 12Hz tokenizer

        # NOTE: Neuron-traced speech decoder is disabled. The speech decoder
        # architecture (sliding window attention, DynamicCache, SnakeBeta activations,
        # causal convolutions) does not trace correctly with torch_neuronx.trace()
        # and produces noise/garbage output. CPU decode with multi-threading is used
        # instead and achieves excellent RTF (~0.69).

        # Always load CPU speech tokenizer as fallback for VQ decode
        try:
            sys.path.insert(0, "/home/ubuntu/Qwen3-TTS")
            from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
            self.speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
                self.speech_tokenizer_dir
            )
            print(f"[Pipeline] Speech tokenizer loaded from {self.speech_tokenizer_dir}")
        except Exception as e:
            print(f"[Pipeline] WARNING: Could not load speech tokenizer via qwen_tts: {e}")
            # Fallback: try direct loading
            try:
                from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
                from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
                from transformers import AutoConfig, AutoModel, AutoFeatureExtractor

                AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
                AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

                self._speech_feature_extractor = AutoFeatureExtractor.from_pretrained(self.speech_tokenizer_dir)
                self._speech_model = AutoModel.from_pretrained(self.speech_tokenizer_dir)
                self._speech_model.eval()
                print(f"[Pipeline] Speech tokenizer loaded directly from {self.speech_tokenizer_dir}")
            except Exception as e2:
                print(f"[Pipeline] WARNING: Could not load speech tokenizer: {e2}")
                print("[Pipeline] Audio decoding will not be available.")

    def _load_text_tokenizer(self):
        """Load the text tokenizer."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            print("[Pipeline] Text tokenizer loaded")
        except Exception as e:
            print(f"[Pipeline] WARNING: Could not load tokenizer: {e}")
            self.tokenizer = None

    def _warmup(self):
        """Run dummy inputs through all Neuron models to warm up."""
        print("[Pipeline] Warming up Neuron models...")
        for bsize, model in self.neuron_talker_buckets.items():
            dummy = torch.zeros(1, bsize, HIDDEN_SIZE)
            cos = self.rope_cos[:, :, :bsize, :]
            sin = self.rope_sin[:, :, :bsize, :]
            _ = model(dummy, cos, sin)
        # Neuron speech decoder warmup skipped (disabled due to tracing issues)
        print("[Pipeline] Warmup complete")

    def _select_bucket(self, seq_len: int):
        """Find the smallest bucket >= seq_len."""
        for bsize in self.bucket_sizes:
            if bsize >= seq_len:
                return bsize, self.neuron_talker_buckets[bsize]
        return None, None

    def _sample_token(self, logits, do_sample=True, top_k=50, top_p=1.0,
                      temperature=0.9, suppress_tokens=None, repetition_penalty=1.05,
                      past_tokens=None):
        """Sample a token from logits."""
        logits = logits.float()

        # Suppress tokens
        if suppress_tokens is not None:
            logits[suppress_tokens] = float("-inf")

        # Repetition penalty
        if past_tokens is not None and len(past_tokens) > 0 and repetition_penalty != 1.0:
            for t in set(past_tokens):
                if logits[t] > 0:
                    logits[t] /= repetition_penalty
                else:
                    logits[t] *= repetition_penalty

        if not do_sample:
            return logits.argmax().item()

        logits = logits / max(temperature, 1e-6)

        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
            logits[indices_to_remove] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    def _build_prefix_embeddings(self, text: str, speaker: str, language: str,
                                 non_streaming_mode: bool = True):
        """
        Build the prefix input embeddings for the talker.

        Returns:
            prefix_embeds: (1, prefix_len, 2048)
            trailing_text_hidden: (1, trailing_len, 2048) or (1, 1, 2048) for pad
        """
        # Tokenize text
        formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer(formatted_text, return_tensors="pt")["input_ids"]  # (1, seq)

        # Get special embeddings
        special_ids = torch.tensor([[
            self.config["tts_bos_token_id"],  # 151672
            self.config["tts_eos_token_id"],  # 151673
            self.config["tts_pad_token_id"],  # 151671
        ]], dtype=torch.long)
        special_embeds = self.embeddings.text_projection(
            self.embeddings.text_embedding(special_ids)
        )
        tts_bos_embed = special_embeds[:, 0:1, :]  # (1, 1, 2048)
        tts_eos_embed = special_embeds[:, 1:2, :]
        tts_pad_embed = special_embeds[:, 2:3, :]

        # Build codec prefill tokens
        language_id = self.talker_config["codec_language_id"].get(language.lower())
        codec_think_id = self.talker_config["codec_think_id"]
        codec_think_bos_id = self.talker_config["codec_think_bos_id"]
        codec_think_eos_id = self.talker_config["codec_think_eos_id"]
        codec_nothink_id = self.talker_config["codec_nothink_id"]
        codec_pad_id = self.talker_config["codec_pad_id"]
        codec_bos_id = self.talker_config["codec_bos_id"]

        if language_id is not None:
            codec_prefill = [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]
        else:
            codec_prefill = [codec_nothink_id, codec_think_bos_id, codec_think_eos_id]

        codec_prefill_embeds = self.embeddings.codec_embedding(
            torch.tensor([codec_prefill], dtype=torch.long)
        )  # (1, 3or4, 2048)

        codec_suffix_embeds = self.embeddings.codec_embedding(
            torch.tensor([[codec_pad_id, codec_bos_id]], dtype=torch.long)
        )  # (1, 2, 2048)

        # Speaker embedding
        spk_id = self.talker_config["spk_id"].get(speaker.lower())
        if spk_id is not None:
            speaker_embed = self.embeddings.codec_embedding(
                torch.tensor([spk_id], dtype=torch.long)
            ).unsqueeze(0)  # (1, 1, 2048)
            codec_input_embed = torch.cat([codec_prefill_embeds, speaker_embed, codec_suffix_embeds], dim=1)
        else:
            codec_input_embed = torch.cat([codec_prefill_embeds, codec_suffix_embeds], dim=1)

        # Role tokens: <|im_start|>assistant\n = first 3 tokens of input_ids
        role_embed = self.embeddings.text_projection(
            self.embeddings.text_embedding(input_ids[:, :3])
        )  # (1, 3, 2048)

        # Pad + bos overlay on codec (without last codec token)
        num_codec_prefix = codec_input_embed.shape[1] - 2  # exclude last 2 (pad, bos)
        pad_bos_embed = torch.cat([
            tts_pad_embed.expand(-1, codec_input_embed.shape[1] - 2, -1),
            tts_bos_embed,
        ], dim=1) + codec_input_embed[:, :-1, :]

        talker_input = torch.cat([role_embed, pad_bos_embed], dim=1)

        if non_streaming_mode:
            # Non-streaming: embed all text + eos, overlay with codec_pad
            text_content_ids = input_ids[:, 3:-5]  # Remove role prefix and suffix tokens
            text_content_embeds = self.embeddings.text_projection(
                self.embeddings.text_embedding(text_content_ids)
            )
            text_with_eos = torch.cat([text_content_embeds, tts_eos_embed], dim=1)
            text_len = text_with_eos.shape[1]

            codec_pad_overlay = self.embeddings.codec_embedding(
                torch.tensor([[codec_pad_id] * text_len], dtype=torch.long)
            )
            text_block = text_with_eos + codec_pad_overlay

            # Final codec_bos token
            codec_bos_final = tts_pad_embed + self.embeddings.codec_embedding(
                torch.tensor([[codec_bos_id]], dtype=torch.long)
            )

            # Append full text block + codec_bos after the complete codec prefix overlay.
            # (Reference first adds then removes first_text+codec_bos; we skip that
            # intermediate step and directly append the text block.)
            talker_input = torch.cat([talker_input, text_block, codec_bos_final], dim=1)
            trailing_text_hidden = tts_pad_embed
        else:
            # Streaming mode
            first_text_embed = self.embeddings.text_projection(
                self.embeddings.text_embedding(input_ids[:, 3:4])
            ) + codec_input_embed[:, -1:, :]
            talker_input = torch.cat([talker_input, first_text_embed], dim=1)

            trailing_text_embeds = self.embeddings.text_projection(
                self.embeddings.text_embedding(input_ids[:, 4:-5])
            )
            trailing_text_hidden = torch.cat([trailing_text_embeds, tts_eos_embed], dim=1)

        return talker_input, trailing_text_hidden, tts_pad_embed

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: str = "Ryan",
        language: str = "English",
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_do_sample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        non_streaming_mode: bool = True,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech from text.

        Returns:
            wavs: list of 1-D float32 numpy arrays
            sample_rate: int
        """
        # Use single thread for generation loop (Neuron does heavy lifting)
        torch.set_num_threads(1)

        codec_eos_token_id = self.talker_config["codec_eos_token_id"]

        # Build suppress_tokens: tokens in range [vocab-1024, vocab) except eos
        suppress_tokens = [
            i for i in range(VOCAB_SIZE - 1024, VOCAB_SIZE)
            if i != codec_eos_token_id
        ]

        # Build prefix embeddings
        prefix_embeds, trailing_text_hidden, tts_pad_embed = self._build_prefix_embeddings(
            text, speaker, language, non_streaming_mode
        )

        prefix_len = prefix_embeds.shape[1]
        print(f"[Generate] Prefix length: {prefix_len}, max_new_tokens: {max_new_tokens}")

        # Pre-allocate buffer
        buffer = torch.zeros(1, self.max_seq_len, HIDDEN_SIZE)
        buffer[:, :prefix_len, :] = prefix_embeds

        seq_len = prefix_len
        all_codec_tokens = []  # List of (16,) code tuples
        past_first_codes = []  # For repetition penalty

        start_time = time.time()
        ttfa_time = None  # Time to first audio token
        talker_total = 0.0
        cp_total = 0.0
        embed_total = 0.0

        for step in range(max_new_tokens):
            # Select bucket
            bsize, neuron_model = self._select_bucket(seq_len)
            if bsize is None:
                print(f"[Generate] Sequence length {seq_len} exceeds max bucket. Stopping.")
                break

            # Run Neuron talker
            t0 = time.time()
            input_slice = buffer[:, :bsize, :]
            cos = self.rope_cos[:, :, :bsize, :]
            sin = self.rope_sin[:, :, :bsize, :]
            hidden_states = neuron_model(input_slice, cos, sin)
            talker_total += time.time() - t0

            # Extract last hidden state at current position
            last_hidden = hidden_states[:, seq_len - 1:seq_len, :]  # (1, 1, 2048)

            # Get codec logits for codebook 0
            logits = self.embeddings.get_codec_logits(last_hidden).squeeze(0).squeeze(0)  # (3072,)

            # Sample first codebook token
            first_code = self._sample_token(
                logits, do_sample, top_k, top_p, temperature,
                suppress_tokens, repetition_penalty, past_first_codes
            )

            # Check for EOS
            if first_code == codec_eos_token_id:
                print(f"[Generate] EOS at step {step}, seq_len={seq_len}")
                break

            past_first_codes.append(first_code)

            # Measure TTFA: time from start to first codec token
            if ttfa_time is None:
                ttfa_time = time.time() - start_time

            # Run code predictor to get codebooks 1-15
            # The CP needs: talker hidden + first code embedding (from talker's codec_embedding)
            t0 = time.time()
            first_code_embed = self.embeddings.codec_embedding(
                torch.tensor([first_code], dtype=torch.long)
            )  # (1, 2048)
            all_codes, cp_embeds_sum = self.code_predictor.generate(
                last_hidden,
                first_code,
                first_code_embed,
                do_sample=subtalker_do_sample,
                top_k=subtalker_top_k,
                top_p=subtalker_top_p,
                temperature=subtalker_temperature,
            )
            cp_total += time.time() - t0

            all_codec_tokens.append(all_codes)

            # Compute next input embedding:
            # Sum of all 16 codebook embeddings + trailing text (or pad)
            t0 = time.time()
            total_embed = first_code_embed + cp_embeds_sum  # (1, 2048) — reuse first_code_embed from above
            total_embed = total_embed.unsqueeze(0)  # (1, 1, 2048)

            # Add trailing text hidden
            generation_step = step
            if generation_step < trailing_text_hidden.shape[1]:
                total_embed = total_embed + trailing_text_hidden[:, generation_step:generation_step + 1, :]
            else:
                total_embed = total_embed + tts_pad_embed

            # Write to buffer
            if seq_len < self.max_seq_len:
                buffer[:, seq_len:seq_len + 1, :] = total_embed
                seq_len += 1
            else:
                print(f"[Generate] Buffer full at step {step}")
                embed_total += time.time() - t0
                break
            embed_total += time.time() - t0

        elapsed = time.time() - start_time
        num_steps = len(all_codec_tokens)
        if num_steps > 0:
            print(f"[Generate] Generated {num_steps} codec steps in {elapsed:.2f}s "
                  f"({num_steps / elapsed:.1f} steps/s)")
            print(f"[Generate] TTFA: {ttfa_time * 1000:.1f}ms")
            print(f"[Generate] Talker: {talker_total:.2f}s ({talker_total/num_steps*1000:.1f}ms/step), "
                  f"CP: {cp_total:.2f}s ({cp_total/num_steps*1000:.1f}ms/step), "
                  f"Embed: {embed_total:.2f}s ({embed_total/num_steps*1000:.1f}ms/step)")

        # Convert codec tokens to audio
        if num_steps == 0:
            return [np.zeros(0, dtype=np.float32)], 24000

        # Stack codes: (num_steps, 16)
        codes_tensor = torch.tensor(all_codec_tokens, dtype=torch.long)  # (T, 16)

        if self.speech_tokenizer is not None or hasattr(self, '_speech_model'):
            # Use multiple CPU threads for the decoder transformer + convolutions
            import multiprocessing
            num_cpus = min(multiprocessing.cpu_count(), 8)
            torch.set_num_threads(num_cpus)
            decode_start = time.time()
            wavs, sr = self._decode_codes(codes_tensor)
            decode_elapsed = time.time() - decode_start
            print(f"[Generate] Speech decode: {decode_elapsed:.2f}s (threads={num_cpus})")
            torch.set_num_threads(1)  # Reset for next generation call
            return wavs, sr
        else:
            print("[Generate] No speech tokenizer available. Returning codes.")
            return codes_tensor, 0

    @torch.inference_mode()
    def generate_streaming(
        self,
        text: str,
        speaker: str = "Ryan",
        language: str = "English",
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_do_sample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        non_streaming_mode: bool = True,
        first_chunk_tokens: int = 6,
        chunk_tokens: int = 12,
    ) -> Generator[Tuple[List[list], float, dict], None, None]:
        """
        Streaming speech generation — yields codec token batches as they are produced.

        Each yield is a tuple of:
            - codes_batch: list of (16,) code tuples for this chunk
            - ttfa_time: time to first codec token (set on first yield, 0.0 after)
            - info: dict with timing/state info (set on final yield)

        The caller is responsible for decoding codec tokens to audio via
        decode_codes_chunk().
        """
        torch.set_num_threads(1)

        codec_eos_token_id = self.talker_config["codec_eos_token_id"]
        suppress_tokens = [
            i for i in range(VOCAB_SIZE - 1024, VOCAB_SIZE)
            if i != codec_eos_token_id
        ]

        prefix_embeds, trailing_text_hidden, tts_pad_embed = self._build_prefix_embeddings(
            text, speaker, language, non_streaming_mode
        )

        prefix_len = prefix_embeds.shape[1]
        print(f"[GenerateStream] Prefix length: {prefix_len}, max_new_tokens: {max_new_tokens}")

        buffer = torch.zeros(1, self.max_seq_len, HIDDEN_SIZE)
        buffer[:, :prefix_len, :] = prefix_embeds

        seq_len = prefix_len
        all_codec_tokens = []
        pending_tokens = []
        past_first_codes = []
        total_yielded = 0

        start_time = time.time()
        ttfa_time = None
        talker_total = 0.0
        cp_total = 0.0
        embed_total = 0.0

        for step in range(max_new_tokens):
            bsize, neuron_model = self._select_bucket(seq_len)
            if bsize is None:
                print(f"[GenerateStream] Sequence length {seq_len} exceeds max bucket. Stopping.")
                break

            t0 = time.time()
            input_slice = buffer[:, :bsize, :]
            cos = self.rope_cos[:, :, :bsize, :]
            sin = self.rope_sin[:, :, :bsize, :]
            hidden_states = neuron_model(input_slice, cos, sin)
            talker_total += time.time() - t0

            last_hidden = hidden_states[:, seq_len - 1:seq_len, :]
            logits = self.embeddings.get_codec_logits(last_hidden).squeeze(0).squeeze(0)

            first_code = self._sample_token(
                logits, do_sample, top_k, top_p, temperature,
                suppress_tokens, repetition_penalty, past_first_codes
            )

            if first_code == codec_eos_token_id:
                print(f"[GenerateStream] EOS at step {step}, seq_len={seq_len}")
                break

            past_first_codes.append(first_code)

            if ttfa_time is None:
                ttfa_time = time.time() - start_time

            t0 = time.time()
            first_code_embed = self.embeddings.codec_embedding(
                torch.tensor([first_code], dtype=torch.long)
            )
            all_codes, cp_embeds_sum = self.code_predictor.generate(
                last_hidden,
                first_code,
                first_code_embed,
                do_sample=subtalker_do_sample,
                top_k=subtalker_top_k,
                top_p=subtalker_top_p,
                temperature=subtalker_temperature,
            )
            cp_total += time.time() - t0

            all_codec_tokens.append(all_codes)
            pending_tokens.append(all_codes)

            t0 = time.time()
            total_embed = first_code_embed + cp_embeds_sum
            total_embed = total_embed.unsqueeze(0)

            generation_step = step
            if generation_step < trailing_text_hidden.shape[1]:
                total_embed = total_embed + trailing_text_hidden[:, generation_step:generation_step + 1, :]
            else:
                total_embed = total_embed + tts_pad_embed

            if seq_len < self.max_seq_len:
                buffer[:, seq_len:seq_len + 1, :] = total_embed
                seq_len += 1
            else:
                print(f"[GenerateStream] Buffer full at step {step}")
                embed_total += time.time() - t0
                break
            embed_total += time.time() - t0

            # Determine if we should yield this batch
            threshold = first_chunk_tokens if total_yielded == 0 else chunk_tokens
            if len(pending_tokens) >= threshold:
                batch = list(pending_tokens)
                pending_tokens.clear()
                report_ttfa = ttfa_time if total_yielded == 0 else 0.0
                total_yielded += len(batch)
                yield batch, report_ttfa, {}

        # Yield any remaining tokens
        if pending_tokens:
            batch = list(pending_tokens)
            pending_tokens.clear()
            report_ttfa = ttfa_time if total_yielded == 0 else 0.0
            total_yielded += len(batch)

        elapsed = time.time() - start_time
        num_steps = len(all_codec_tokens)
        info = {
            "num_steps": num_steps,
            "elapsed": elapsed,
            "ttfa": ttfa_time or 0.0,
            "talker_total": talker_total,
            "cp_total": cp_total,
            "embed_total": embed_total,
        }
        if num_steps > 0:
            print(f"[GenerateStream] Generated {num_steps} codec steps in {elapsed:.2f}s "
                  f"({num_steps / elapsed:.1f} steps/s)")
            print(f"[GenerateStream] TTFA: {(ttfa_time or 0) * 1000:.1f}ms")
            print(f"[GenerateStream] Talker: {talker_total:.2f}s ({talker_total/num_steps*1000:.1f}ms/step), "
                  f"CP: {cp_total:.2f}s ({cp_total/num_steps*1000:.1f}ms/step), "
                  f"Embed: {embed_total:.2f}s ({embed_total/num_steps*1000:.1f}ms/step)")

        if pending_tokens or (total_yielded == 0 and num_steps > 0):
            # Final yield with remaining tokens (if any) or empty with info
            remaining = list(pending_tokens) if pending_tokens else []
            yield remaining, report_ttfa if total_yielded == 0 else 0.0, info
        else:
            # Yield empty batch just to deliver info
            yield [], 0.0, info

    def decode_codes_chunk(
        self,
        codes: List[list],
        left_context: List[list] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Decode a chunk of codec tokens to audio.

        Args:
            codes: list of (16,) code tuples to decode
            left_context: optional list of preceding (16,) code tuples for decoder context

        Returns:
            wav: 1-D float32 numpy array (audio for this chunk, excluding context)
            sr: sample rate (24000)
        """
        import multiprocessing
        sr = 24000

        if not codes and not left_context:
            return np.zeros(0, dtype=np.float32), sr

        # Build full sequence: context + new codes
        full_codes = list(left_context or []) + list(codes)
        if not full_codes:
            return np.zeros(0, dtype=np.float32), sr

        codes_tensor = torch.tensor(full_codes, dtype=torch.long)

        num_cpus = min(multiprocessing.cpu_count(), 8)
        torch.set_num_threads(num_cpus)

        if self.speech_tokenizer is not None:
            encoded = {"audio_codes": [codes_tensor]}
            wavs, sr = self.speech_tokenizer.decode(encoded)
        elif hasattr(self, '_speech_model'):
            ct = codes_tensor.unsqueeze(0)
            with torch.inference_mode():
                dec = self._speech_model.decode(ct, return_dict=True)
                wav_tensors = dec.audio_values
            wavs = [w.to(torch.float32).detach().cpu().numpy() for w in wav_tensors]
            sr = int(self._speech_model.get_output_sample_rate())
        else:
            torch.set_num_threads(1)
            raise RuntimeError("No speech tokenizer available for decoding")

        torch.set_num_threads(1)

        wav = wavs[0] if wavs else np.zeros(0, dtype=np.float32)
        if isinstance(wav, torch.Tensor):
            wav = wav.float().cpu().numpy()

        # Trim left context from output
        if left_context:
            ctx_samples = len(left_context) * self._speech_decode_upsample
            wav = wav[ctx_samples:]

        return wav, sr

    def _decode_codes(self, codes_tensor: torch.Tensor) -> Tuple[List[np.ndarray], int]:
        """Decode codec tokens to audio using the speech tokenizer.

        Prefers Neuron-traced decoder when available; falls back to CPU.
        """
        # codes_tensor: (T, 16) where T is number of time steps
        sr = 24000

        # NOTE: Neuron-traced speech decoder is disabled — the speech decoder
        # architecture (sliding window attention, DynamicCache, SnakeBeta activations,
        # causal convolutions) does not trace correctly and produces noise/garbage.
        # CPU decode with multi-threading achieves RTF ~0.69 which exceeds targets.

        if self.speech_tokenizer is not None:
            # Use the QwenTTS tokenizer wrapper (CPU)
            encoded = {"audio_codes": [codes_tensor]}
            wavs, sr = self.speech_tokenizer.decode(encoded)
            return wavs, sr
        elif hasattr(self, '_speech_model'):
            codes = codes_tensor.unsqueeze(0)  # (1, T, 16)
            with torch.inference_mode():
                dec = self._speech_model.decode(codes, return_dict=True)
                wav_tensors = dec.audio_values
            wavs = [w.to(torch.float32).detach().cpu().numpy() for w in wav_tensors]
            sr = int(self._speech_model.get_output_sample_rate())
            return wavs, sr
        else:
            raise RuntimeError("No speech tokenizer available for decoding")

    @torch.inference_mode()
    def _decode_codes_neuron(self, codes_tensor: torch.Tensor) -> List[np.ndarray]:
        """Decode using Neuron-traced speech decoder with chunked processing.

        The traced model expects fixed input shape (1, 16, chunk_size).
        We process codes in chunks, including left context within the
        fixed chunk_size budget.
        """
        # codes_tensor: (T, 16) -> transpose to (1, 16, T) for the decoder
        codes = codes_tensor.T.unsqueeze(0).long()  # (1, 16, T)
        T = codes.shape[2]

        chunk_size = self._neuron_speech_chunk_size
        left_context = 25
        # Effective new data per chunk = chunk_size - left_context
        stride = chunk_size - left_context
        wavs = []
        start = 0

        while start < T:
            end = min(start + stride, T)
            ctx = min(left_context, start)
            chunk = codes[:, :, start - ctx: end]

            # Pad to exactly chunk_size
            actual_len = chunk.shape[2]
            if actual_len < chunk_size:
                pad_amount = chunk_size - actual_len
                chunk = F.pad(chunk, (0, pad_amount), value=0)
            elif actual_len > chunk_size:
                # Should not happen with correct stride, but truncate to be safe
                chunk = chunk[:, :, :chunk_size]
                end = start + chunk_size - ctx

            wav_chunk = self._neuron_speech_decoder(chunk)

            # Trim context and padding from output
            ctx_samples = ctx * self._speech_decode_upsample
            valid_len = (end - start) * self._speech_decode_upsample
            wav_valid = wav_chunk[0, 0, ctx_samples: ctx_samples + valid_len]
            wavs.append(wav_valid)

            start = end

        full_wav = torch.cat(wavs, dim=0).float().cpu().numpy()
        return [full_wav]

    def get_supported_speakers(self) -> list:
        """Return list of supported speaker names."""
        return list(self.talker_config.get("spk_id", {}).keys())

    def get_supported_languages(self) -> list:
        """Return list of supported languages."""
        return list(self.talker_config.get("codec_language_id", {}).keys())
