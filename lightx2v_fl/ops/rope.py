"""Rotary position embedding for FlagOS via FlagGems with a torch fallback."""

from __future__ import annotations

import warnings
import torch

from lightx2v.utils.registry_factory import ROPE_REGISTER
from lightx2v_platform.ops.rope.rope_template import RopeTemplate

try:
    import flag_gems
    _GEMS_ROPE = getattr(flag_gems, "apply_rotary_pos_emb", None)
except Exception:  # pragma: no cover
    flag_gems = None
    _GEMS_ROPE = None

_ROPE_STATS = {"gems": 0, "fallback": 0}


def _apply_rotary_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved rotary application in pure torch.

    x: [..., D] with D even; cos/sin: [..., D/2] broadcastable to x's leading dims.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot_even = x1 * cos - x2 * sin
    rot_odd = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


@ROPE_REGISTER("flagos_rope")
class FlagOSRope(RopeTemplate):
    def apply(self, xq: torch.Tensor, xk: torch.Tensor, cos_sin_cache: torch.Tensor):
        # LightX2V: xq/xk [S, H, D]
        if xq.ndim != 3 or xk.ndim != 3:
            raise RuntimeError(f"flagos_rope expects xq/xk [S,H,D], got {xq.shape}, {xk.shape}")

        if torch.is_complex(cos_sin_cache):
            cos = cos_sin_cache.real.contiguous()
            sin = cos_sin_cache.imag.contiguous()
        else:
            half = cos_sin_cache.shape[-1] // 2
            cos = cos_sin_cache[..., :half].contiguous()
            sin = cos_sin_cache[..., half:].contiguous()

        # --- 尝试使用 FlagGems 算子 ---
        if _GEMS_ROPE is not None and callable(_GEMS_ROPE):
            try:
                # 【关键安全改动】仅在 FlagGems 内部进行特定维度的 squeeze
                # 这样如果报错进入 fallback，外部的 cos/sin 依然保持原版布局 [S, 1, D/2]，防止 PyTorch 广播失败
                gems_cos = cos.squeeze(1).contiguous() if (cos.ndim == 3 and cos.shape[1] == 1) else cos
                gems_sin = sin.squeeze(1).contiguous() if (sin.ndim == 3 and sin.shape[1] == 1) else sin

                # 适配 FlagGems 的 [B, H, S, D] 布局 (这里 B=1)
                xq_bhsd = xq.permute(1, 0, 2).unsqueeze(0).contiguous()
                xk_bhsd = xk.permute(1, 0, 2).unsqueeze(0).contiguous()

                xq_o, xk_o = _GEMS_ROPE(
                    xq_bhsd,
                    xk_bhsd,
                    gems_cos,
                    gems_sin,
                    rotary_interleaved=True,
                )

                # 还原回 LightX2V 的 [S, H, D] 布局
                xq_o = xq_o.squeeze(0).permute(1, 0, 2).contiguous()
                xk_o = xk_o.squeeze(0).permute(1, 0, 2).contiguous()

                _ROPE_STATS["gems"] += 1
                if _ROPE_STATS["gems"] <= 5:
                    print(
                        "[flagos_rope_runtime] backend=FlagGems",
                        "xq=", tuple(xq.shape),
                        "xk=", tuple(xk.shape),
                        "cos=", tuple(gems_cos.shape),
                        flush=True,
                    )
                return xq_o.to(self.infer_dtype), xk_o.to(self.infer_dtype)
                
            except Exception as exc:  # pragma: no cover
                # 捕获运行期算子异常，打印警告，静默准备进入 Fallback
                warnings.warn(f"[flagos_rope] FlagGems RoPE failed ({exc}); falling back to torch implementation.", stacklevel=2)

        # --- Fallback: 使用原生的纯 PyTorch 实现 ---
        _ROPE_STATS["fallback"] += 1
        if _ROPE_STATS["fallback"] <= 5:
            print(
                "[flagos_rope_runtime] backend=PyTorch (Fallback)",
                "xq=", tuple(xq.shape),
                "xk=", tuple(xk.shape),
                flush=True,
            )

        xq_o = _apply_rotary_ref(xq, cos, sin)
        xk_o = _apply_rotary_ref(xk, cos, sin)
        return xq_o.to(self.infer_dtype), xk_o.to(self.infer_dtype)