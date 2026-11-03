"""Optional global aten-level patching with FlagGems.

LightX2V routes its heavy linear/attn/norm ops through weight templates (which
we override in ``lightx2v_fl.ops``). But a model still issues many *generic*
torch ops — softmax, elementwise, reductions — directly. ``flag_gems.enable()``
lowers those to FlagGems Triton kernels at the aten dispatch level, giving
cross-chip coverage for the long tail too.

This is opt-in (LIGHTX2V_FL_GLOBAL_GEMS=1) because it changes global torch
behaviour for the whole process. The template path is the primary, always-on
mechanism; this stacks on top.
"""

from __future__ import annotations

from loguru import logger

from .aten_policy import get_global_gems_lists

_ENABLED = False


def enable_global_flaggems(
    *, model_name: str | None = None, vendor_name: str | None = None
) -> None:
    global _ENABLED
    if _ENABLED:
        return
    try:
        import flag_gems
    except Exception as exc:  # noqa: BLE001 - preserve optional vendor import behavior
        logger.warning(f"[lightx2v-fl] cannot enable global FlagGems: {exc}")
        return

    whitelist, blacklist = get_global_gems_lists(
        vendor_name=vendor_name or getattr(flag_gems, "vendor_name", "unknown"),
        model_name=model_name,
    )
    if whitelist is not None:
        only_enable = getattr(flag_gems, "only_enable", None)
        if not callable(only_enable):
            raise RuntimeError(
                "This FlagGems build does not support only_enable(include=...)."
            )
        only_enable(include=whitelist)
    else:
        flag_gems.enable(unused=blacklist)
    _ENABLED = True
    logger.info("[lightx2v-fl] global FlagGems aten patching enabled.")
