# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NPU patches for the MiniMax H3 Qwen3-VL text encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from vllm.logger import init_logger
from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
from vllm_ascend.device.device_config import is_310p
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, aligned_16, nd_to_nz_2d

from vllm_omni.platforms.npu.layers.rotary_embedding import (
    npu_rotary_mul_with_bsnd_fallback,
)

logger = init_logger(__name__)

_ROPE_PATCHED = False
_SDPA_PATCHED = False
_SWIGLU_PATCHED = False
_CAUSAL_MASK_310P: dict[tuple[torch.device, int], torch.Tensor] = {}


def _causal_mask_310p(device: torch.device, sequence_length: int) -> torch.Tensor:
    key = (device, sequence_length)
    mask = _CAUSAL_MASK_310P.get(key)
    if mask is None:
        mask = AttentionMaskBuilder310.gen_causal_additive_mask(sequence_length, device)
        mask = torch_npu.npu_format_cast(nd_to_nz_2d(mask), ACL_FORMAT_FRACTAL_NZ)
        _CAUSAL_MASK_310P[key] = mask
    return mask


def _scaled_dot_product_attention_310p(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Run causal self-attention through 310P native flash attention."""
    batch_size, num_heads, sequence_length, head_dim = query.shape
    num_key_value_heads = key.shape[1]
    real_tokens = batch_size * sequence_length

    query_flat = aligned_16(query.transpose(1, 2).reshape(real_tokens, num_heads, head_dim))
    key_flat = aligned_16(key.transpose(1, 2).reshape(real_tokens, num_key_value_heads, head_dim))
    value_flat = aligned_16(value.transpose(1, 2).reshape(real_tokens, num_key_value_heads, head_dim))
    aligned_tokens = int(query_flat.shape[0])

    sequence_lengths = torch.full((batch_size,), sequence_length, dtype=torch.int32, device="cpu")
    if aligned_tokens > real_tokens:
        sequence_lengths[-1] += aligned_tokens - real_tokens
    mask_length = int(sequence_lengths.max().item())
    mask = _causal_mask_310p(query.device, mask_length)
    output = torch.empty(
        (aligned_tokens, num_heads, head_dim),
        dtype=torch.float16,
        device=query.device,
    )
    torch_npu._npu_flash_attention(
        query=query_flat.contiguous(),
        key=key_flat.contiguous(),
        value=value_flat.contiguous(),
        mask=mask,
        seq_len=sequence_lengths,
        scale_value=head_dim**-0.5,
        num_heads=num_heads,
        num_kv_heads=num_key_value_heads,
        out=output,
    )
    return output[:real_tokens].reshape(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)


def _apply_rotary_pos_emb_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3-VL text RoPE with fused BNSD rotary multiplication."""
    return (
        npu_rotary_mul_with_bsnd_fallback(q, cos, sin, unsqueeze_dim=1),
        npu_rotary_mul_with_bsnd_fallback(k, cos, sin, unsqueeze_dim=1),
    )


def apply_minimax_h3_qwen3vl_patch() -> None:
    """Route MiniMax H3 Qwen3-VL text RoPE to the Ascend fused operator."""
    global _ROPE_PATCHED
    if _ROPE_PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder._apply_rotary_pos_emb = _apply_rotary_pos_emb_npu
    _ROPE_PATCHED = True
    logger.debug("Applied NPU fused RoPE patch for MiniMax H3 Qwen3-VL text encoder")


def _scaled_dot_product_attention_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Run causal SDPA with compressed K/V heads through NPU native GQA."""
    num_heads = query.shape[1]
    num_key_value_heads = key.shape[1]
    num_value_heads = value.shape[1]
    if num_key_value_heads != num_value_heads:
        raise ValueError(
            "GQA requires key and value to have the same number of heads, "
            f"got k_heads={num_key_value_heads} and v_heads={num_value_heads}."
        )
    if num_key_value_heads == 0:
        raise ValueError("GQA requires at least one KV head.")
    if num_heads % num_key_value_heads != 0:
        raise ValueError(
            "GQA requires query heads to be a multiple of KV heads, "
            f"got q_heads={num_heads} and kv_heads={num_key_value_heads}."
        )
    if is_310p():
        return _scaled_dot_product_attention_310p(query, key, value)

    return F.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        dropout_p=0.0,
        is_causal=True,
        enable_gqa=num_heads != num_key_value_heads,
    )


def _scaled_dot_product_attention_vision_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    if not is_310p():
        return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    batch_size, num_heads, sequence_length, head_dim = query.shape
    q = query.transpose(1, 2).reshape(-1, num_heads, head_dim).to(torch.float16).contiguous()
    k = key.transpose(1, 2).reshape(-1, key.shape[1], head_dim).to(torch.float16).contiguous()
    v = value.transpose(1, 2).reshape(-1, value.shape[1], head_dim).to(torch.float16).contiguous()
    output = torch.empty_like(q)
    sequence_lengths = torch.full((batch_size,), sequence_length, dtype=torch.int32, device="cpu")
    torch_npu._npu_flash_attention_unpad(
        query=q,
        key=k,
        value=v,
        seq_len=sequence_lengths,
        scale_value=head_dim**-0.5,
        num_heads=num_heads,
        num_kv_heads=key.shape[1],
        out=output,
    )
    return output.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2).to(query.dtype)


def apply_minimax_h3_qwen3vl_sdpa_patch() -> None:
    """Route MiniMax H3 Qwen3-VL text attention to NPU native GQA."""
    global _SDPA_PATCHED
    if _SDPA_PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder._scaled_dot_product_attention = _scaled_dot_product_attention_npu
    encoder._scaled_dot_product_attention_vision = _scaled_dot_product_attention_vision_npu
    _SDPA_PATCHED = True
    logger.debug("Applied NPU SDPA patch for MiniMax H3 Qwen3-VL text encoder")


def npu_swiglu_from_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply fused SwiGLU to a packed gate/up projection tensor."""
    return torch_npu.npu_swiglu(gate_up, dim=-1)


def _forward_minimax_h3_qwen3vl_text_mlp_npu(self: Any, x: torch.Tensor) -> torch.Tensor:
    """Run the Qwen3-VL MLP with one packed GEMM and fused SwiGLU."""
    gate_up = F.linear(x, self.gate_up_proj.weight)
    return self.down_proj(npu_swiglu_from_packed(gate_up))


def apply_minimax_h3_qwen3vl_swiglu_patch() -> None:
    """Route MiniMax H3 Qwen3-VL text MLP to the Ascend fused SwiGLU path."""
    global _SWIGLU_PATCHED
    if _SWIGLU_PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder.MiniMaxH3Qwen3VLTextMLP.forward = _forward_minimax_h3_qwen3vl_text_mlp_npu  # type: ignore[method-assign]
    _SWIGLU_PATCHED = True
