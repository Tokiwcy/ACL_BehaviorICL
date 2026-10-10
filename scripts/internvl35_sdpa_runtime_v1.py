"""Bound native SDPA kernels for Windows: cuDNN prefill, math one-token decode."""
from __future__ import annotations

NAME = "internvl_cudnn_prefill_math_decode"


def attention(module, query, key, value, attention_mask, **kwargs):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    backend = SDPBackend.CUDNN_ATTENTION if query.shape[2] > 1 else SDPBackend.MATH
    with sdpa_kernel(backend):
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def register_backend():
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    AttentionInterface.register(NAME, attention)
    AttentionMaskInterface.register(NAME, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
