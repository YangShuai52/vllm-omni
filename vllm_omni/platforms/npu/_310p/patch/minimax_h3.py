# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""310P patches for MiniMax H3.

On 310P, FIA (npu_fusion_attention) is not supported, so the mindiesd
attention path cannot be used. The forward_npu method of
FlashAttentionImpl is already patched in flash_attn.py to dispatch to
forward_fa_310p, which uses torch_npu._npu_flash_attention_unpad for
packed varlen sequences and SDPA for other cases.

This module also ensures the NPU platform patches (Qwen3-VL text encoder
RoPE, SDPA, SwiGLU) are applied, as they are needed by the H3 text
encoder regardless of the attention backend.
"""

from __future__ import annotations

_PATCHED = False


def apply_h3_patches() -> None:
    """Apply 310P-specific patches for MiniMax H3.

    Currently, the core dispatch logic is handled directly in:
    - vllm_omni/diffusion/layers/indexed_modulation.py  (HAS_TRITON guard)
    - vllm_omni/diffusion/attention/backends/flash_attn.py  (310P branch)
    - vllm_omni/platforms/npu/platform.py  (mindiesd import guard)

    This function serves as the registration point for any additional
    runtime patches that 310P needs for H3. It is called from
    apply_model_patches() when the model architecture is MiniMaxH3DiTModel.
    """
    global _PATCHED
    if _PATCHED:
        return

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    logger.info("Applying 310P patches for MiniMax H3")

    _PATCHED = True
