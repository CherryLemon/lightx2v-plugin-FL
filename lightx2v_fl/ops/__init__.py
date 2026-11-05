"""
FlagOS operator implementations for LightX2V.

Each module defines a weight-template subclass whose ``apply()`` calls into
FlagGems, with a torch reference fallback when FlagGems is unavailable or does
not cover the op. Registration targets the **final** LightX2V registries
(``lightx2v.utils.registry_factory``) rather than the ``PLATFORM_*`` staging
tables, so it is robust to the one-shot ``merge`` snapshot regardless of import
order. See ``lightx2v_fl.register`` for the timing rationale.

Registry keys (use these as the ``*_type`` / ``dit_quant_scheme`` values in a
config JSON):
    self_attn_1_type / cross_attn_1_type / cross_attn_2_type : "flagos_flash_attn"
    rms_norm_type                                            : "flagos_rms_norm"
    layer_norm_type                                          : "flagos_layer_norm"
    rope_type                                                : "flagos_rope"
    dit_quant_scheme (mm)                                    : "flagos" (bf16/fp16),
                                                               "flagos-fp8",
                                                               "flagos-int8"
"""

from __future__ import annotations

from loguru import logger


def register_ops() -> None:
    """Import every op module for its registration side effects."""
    # Importing each module runs its @*_REGISTER decorators against the final
    # lightx2v registries.
    from . import attn  # noqa: F401
    from . import h3_norm  # noqa: F401
    from . import mm  # noqa: F401
    from . import norm  # noqa: F401
    from . import rope  # noqa: F401
    from . import sage_attn  # noqa: F401

    logger.debug("[lightx2v-fl] flagos ops registered.")
