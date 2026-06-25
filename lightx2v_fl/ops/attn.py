"""Flash-attention op backed by FlagGems, with a torch SDPA fallback.

Layout follows the WAN varlen convention used across lightx2v_platform:
    q / k / v : [S, H, D]  (or [B, S, H, D])  where S = total tokens
    output    : [S, H*D]
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.ops.attn.template import AttnWeightTemplate


def _resolve_flaggems_attention():
    """尝试多入口深度探测 FlagGems attention 算子"""
    try:
        import flag_gems
    except Exception:
        return None

    attn_obj = getattr(getattr(flag_gems, "ops", None), "attention", None)

    # 1. 如果 ops.attention 直接可调用
    if callable(attn_obj):
        return attn_obj

    # 2. 如果 ops.attention 是个模块/对象，探测里面的具体函数名
    if attn_obj is not None:
        for name in (
            "scaled_dot_product_attention",
            "flash_attention_forward",
            "flash_attn_varlen_func",
        ):
            fn = getattr(attn_obj, name, None)
            if callable(fn):
                return fn

    # 3. 兼容老版本，直接在 flag_gems 根目录下找
    fn = getattr(flag_gems, "scaled_dot_product_attention", None)
    if callable(fn):
        return fn

    return None


_GEMS_ATTN = _resolve_flaggems_attention()
_ATTN_STATS = {"gems": 0, "fallback": 0}


def _sdp(q4d: torch.Tensor, k4d: torch.Tensor, v4d: torch.Tensor) -> torch.Tensor:
    """Scaled-dot-product attention on [1, L, H, D] inputs → [1, L, H, D]."""
    
    # --- 尝试使用 FlagGems 算子 ---
    if _GEMS_ATTN is not None:
        try:
            q_t = q4d.permute(0, 2, 1, 3).contiguous()
            k_t = k4d.permute(0, 2, 1, 3).contiguous()
            v_t = v4d.permute(0, 2, 1, 3).contiguous()
            
            out = _GEMS_ATTN(q_t, k_t, v_t)
            
            # 统计并打印前 20 次成功调用的日志
            _ATTN_STATS["gems"] += 1
            if _ATTN_STATS["gems"] <= 20:
                print(
                    "[flagos_attn_runtime] backend=FlagGems",
                    "q=", tuple(q4d.shape),
                    "k=", tuple(k4d.shape),
                    "out=", tuple(out.shape),
                    flush=True,
                )
                
            return out.permute(0, 2, 1, 3)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"[flagos_flash_attn] FlagGems attention failed ({exc}); using torch SDPA.", stacklevel=2)

    # --- Fallback: 使用 PyTorch 原生 SDPA ---
    _ATTN_STATS["fallback"] += 1
    if _ATTN_STATS["fallback"] <= 20:
        print(
            "[flagos_attn_runtime] backend=PyTorch (Fallback)",
            "q=", tuple(q4d.shape),
            "k=", tuple(k4d.shape),
            flush=True,
        )

    q_t = q4d.permute(0, 2, 1, 3).contiguous()
    k_t = k4d.permute(0, 2, 1, 3).contiguous()
    v_t = v4d.permute(0, 2, 1, 3).contiguous()
    
    out = F.scaled_dot_product_attention(q_t, k_t, v_t)
    
    return out.permute(0, 2, 1, 3)


@ATTN_WEIGHT_REGISTER("flagos_flash_attn")
class FlagOSFlashAttnWeight(AttnWeightTemplate):
    """FlagOS attention. Registered as ``flagos_flash_attn``."""

    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        if q.ndim == 4:
            bs = q.shape[0]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
        else:
            bs = 1

        total_q = q.shape[0]

        # Fast path: single sequence.
        if cu_seqlens_q is None or bs == 1:
            x = _sdp(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
            return x.squeeze(0).reshape(total_q, -1)

        # Varlen path: one call per sequence (handles differing Q / KV lengths).
        outputs = []
        batch_size = cu_seqlens_q.shape[0] - 1
        ckv = cu_seqlens_kv if cu_seqlens_kv is not None else cu_seqlens_q
        for i in range(batch_size):
            qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
            ks, ke = int(ckv[i]), int(ckv[i + 1])
            xi = _sdp(
                q[qs:qe].unsqueeze(0),
                k[ks:ke].unsqueeze(0),
                v[ks:ke].unsqueeze(0),
            )
            outputs.append(xi.squeeze(0).reshape(qe - qs, -1))
        return torch.cat(outputs, dim=0)