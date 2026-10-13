"""
Flash Attention 2 processor for Qwen double-stream architecture.
Compatible with RTX 5090 and other modern GPUs.
"""

import torch
from typing import Optional, Tuple
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

try:
    from flash_attn import flash_attn_func
    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False
    _flash_attn_err = "flash-attn package not installed"


def _ensure_fa2_available():
    if not _flash_attn_available:
        raise ImportError(
            "FlashAttention-2 via `flash-attn` package is required. "
            f"Install with: pip install flash-attn. Error: {_flash_attn_err}"
        )


class QwenDoubleStreamAttnProcessorFA2:
    """
    FA2-based attention processor for Qwen double-stream architecture.
    Computes joint attention over concatenated [text, image] streams using FlashAttention-2
    via the `flash-attn` package.

    Notes / limitations:
    - General attention masks are not supported here (FA2 path). `is_causal=False` and no arbitrary mask.
    - Expects an available `apply_rotary_emb_qwen` in scope (same as your non-FA2 processor).
    - Compatible with RTX 5090 and other modern GPUs.
    """

    _attention_backend = "fa2"  # for parity with your other processors, not used internally

    def __init__(self):
        _ensure_fa2_available()

    @torch.no_grad()
    def __call__(
        self,
        attn,  # Attention module with to_q/to_k/to_v/add_*_proj, norms, to_out, to_add_out, and .heads
        hidden_states: torch.FloatTensor,                 # (B, S_img, D_model)  image stream
        encoder_hidden_states: torch.FloatTensor = None,  # (B, S_txt, D_model)  text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,  # unused in FA2 path
        attention_mask: Optional[torch.FloatTensor] = None,    # unused in FA2 path
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (img_freqs, txt_freqs)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessorFA2 requires encoder_hidden_states (text stream).")
        if attention_mask is not None:
            # FA2 kernel path here does not consume arbitrary masks; fail fast to avoid silent correctness issues.
            raise NotImplementedError("attention_mask is not supported in this FA2 implementation.")

        _ensure_fa2_available()

        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]

        # ---- QKV projections (image/sample stream) ----
        img_q = attn.to_q(hidden_states)   # (B, S_img, D)
        img_k = attn.to_k(hidden_states)
        img_v = attn.to_v(hidden_states)

        # ---- QKV projections (text/context stream) ----
        txt_q = attn.add_q_proj(encoder_hidden_states)  # (B, S_txt, D)
        txt_k = attn.add_k_proj(encoder_hidden_states)
        txt_v = attn.add_v_proj(encoder_hidden_states)

        # ---- Reshape to (B, S, H, D_h) for flash-attn ----
        H = attn.heads
        img_q = img_q.unflatten(-1, (H, -1))
        img_k = img_k.unflatten(-1, (H, -1))
        img_v = img_v.unflatten(-1, (H, -1))

        txt_q = txt_q.unflatten(-1, (H, -1))
        txt_k = txt_k.unflatten(-1, (H, -1))
        txt_v = txt_v.unflatten(-1, (H, -1))

        # ---- Q/K normalization (per your module contract) ----
        if getattr(attn, "norm_q", None) is not None:
            img_q = attn.norm_q(img_q)
        if getattr(attn, "norm_k", None) is not None:
            img_k = attn.norm_k(img_k)
        if getattr(attn, "norm_added_q", None) is not None:
            txt_q = attn.norm_added_q(txt_q)
        if getattr(attn, "norm_added_k", None) is not None:
            txt_k = attn.norm_added_k(txt_k)

        # ---- RoPE (Qwen variant) ----
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            # expects tensors shaped (B, S, H, D_h)
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, use_real=False)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, use_real=False)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, use_real=False)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, use_real=False)

        # ---- Joint attention over [text, image] along sequence axis ----
        # Shapes: (B, S_total, H, D_h)
        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)

        # FlashAttention-2 path expects (B, S, H, D_h) and returns output tensor
        # flash_attn_func signature: (q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1))
        # Returns: output tensor of shape (B, S_total, H, D_h)
        # Note: flash-attn requires contiguous tensors, but torch.cat already returns contiguous tensors
        out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)  # out: (B, S_total, H, D_h)

        # ---- Back to (B, S, D_model) ----
        out = out.flatten(2, 3).to(q.dtype)

        # Split back to text / image segments
        txt_attn_out = out[:, :S_txt, :]
        img_attn_out = out[:, S_txt:, :]

        # ---- Output projections ----
        img_attn_out = attn.to_out[0](img_attn_out)
        if len(attn.to_out) > 1:
            img_attn_out = attn.to_out[1](img_attn_out)  # dropout if present

        txt_attn_out = attn.to_add_out(txt_attn_out)

        return img_attn_out, txt_attn_out
