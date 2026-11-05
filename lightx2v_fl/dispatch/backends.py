"""Pre-execution adapters to public FlagGems APIs; no numerical kernels."""

from __future__ import annotations

import inspect
from functools import lru_cache

from loguru import logger


@lru_cache(maxsize=1)
def load_flaggems():
    try:
        import flag_gems
    except Exception as exc:  # noqa: BLE001 - optional vendor import, before compute
        logger.debug(f"[lightx2v-fl] FlagGems unavailable for explicit dispatch: {exc}")
        return None
    return flag_gems


def bind_call(fn, args, kwargs):
    """Return a zero-argument call only if its signature accepts the contract."""
    if not callable(fn):
        return None
    try:
        try:
            signature = _signature(fn)
        except TypeError:
            signature = inspect.signature(fn)
        signature.bind(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    return lambda: fn(*args, **kwargs)


@lru_cache(maxsize=128)
def _signature(fn):
    return inspect.signature(fn)


@lru_cache(maxsize=1)
def load_lightx2v_gems():
    try:
        import flaggems_lightx2v
    except ImportError:
        return None
    return flaggems_lightx2v


def lightx2v_call(op_name, args, kwargs):
    library = load_lightx2v_gems()
    if library is None or not op_name.startswith("h3_"):
        return None
    return bind_call(getattr(library, op_name, None), args, kwargs)


def flaggems_call(gems, op_name, args, kwargs):
    if gems is None:
        return None
    if op_name == "sage_attention":
        return bind_call(getattr(gems, "sage_attention", None), args, kwargs)
    if op_name.startswith("h3_"):
        return None
    if op_name == "rms_norm":
        x, weight = args
        fn = getattr(gems, "rms_norm", None)
        # Recognize both public signatures without executing either as a probe.
        return bind_call(fn, (x, (x.shape[-1],), weight), kwargs) or bind_call(
            fn, (x, weight), kwargs
        )
    if op_name == "layer_norm":
        x, weight, bias = args
        if weight is None:
            return None  # Matches the existing non-affine torch path.
        return bind_call(
            getattr(gems, "layer_norm", None),
            (x, (x.shape[-1],), weight, bias),
            kwargs,
        )
    # Only dense SDPA-compatible APIs accept this BHSD -> BHSD contract.
    # Varlen APIs require metadata; flash_attention_forward can return tuples.
    attention = getattr(getattr(gems, "ops", None), "attention", None)
    functions = (
        attention,
        getattr(attention, "scaled_dot_product_attention", None),
        getattr(gems, "scaled_dot_product_attention", None),
    )
    for fn in functions:
        call = bind_call(fn, args, kwargs)
        if call is not None:
            return call
    return None
