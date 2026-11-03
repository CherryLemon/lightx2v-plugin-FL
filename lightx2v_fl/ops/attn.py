"""Flash-attention op backed by FlagGems, with a torch SDPA fallback.

Layout follows the WAN varlen convention used across lightx2v_platform:
    q / k / v : [S, H, D]  (or [B, S, H, D])  where S = total tokens
    output    : [S, H*D]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.ops.attn.template import AttnWeightTemplate

from lightx2v_fl.dispatch import dispatch_op


def _sdp(
    q4d: torch.Tensor, k4d: torch.Tensor, v4d: torch.Tensor, config=None
) -> torch.Tensor:
    """Dense attention on [B, L, H, D] inputs -> [B, L, H, D]."""
    q_t = q4d.permute(0, 2, 1, 3).contiguous()
    k_t = k4d.permute(0, 2, 1, 3).contiguous()
    v_t = v4d.permute(0, 2, 1, 3).contiguous()
    out = dispatch_op(
        "attention",
        q_t,
        k_t,
        v_t,
        torch_impl=_torch_sdp,
        config=config,
    )
    return out.permute(0, 2, 1, 3)


def _torch_sdp(q, k, v, **parameters):
    return F.scaled_dot_product_attention(q, k, v, **parameters)


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
            x = _sdp(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), self.config)
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
                self.config,
            )
            outputs.append(xi.squeeze(0).reshape(qe - qs, -1))
        return torch.cat(outputs, dim=0)
