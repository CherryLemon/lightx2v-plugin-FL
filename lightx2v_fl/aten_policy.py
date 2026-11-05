"""Selection policy for optional process-global FlagGems ATen registration."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

_PREFIX = "LIGHTX2V_FL_GEMS_"


def _read_defaults() -> dict:
    path = Path(__file__).with_name("configs") / "global_gems.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_ops(value: str) -> list[str]:
    return [op.strip() for op in value.split(",") if op.strip()]


def get_global_gems_lists(
    *,
    vendor_name: str = "unknown",
    model_name: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[list[str] | None, list[str] | None]:
    """Return (whitelist, blacklist), following vllm-plugin-FL precedence.

    Raw nonempty strings select a mode; an empty string is unset. Consequently
    a whitespace/comma-only blacklist explicitly clears defaults, whereas such
    a whitelist selects no ops (FlagGems may reject an empty include list).
    UNUSED is a legacy alias for BLACKLIST, not an additional exclusion layer.
    """
    env = os.environ if environ is None else environ
    whitelist = env.get(_PREFIX + "WHITELIST", "")
    blacklist = env.get(_PREFIX + "BLACKLIST", "")
    legacy = env.get(_PREFIX + "UNUSED", "")
    append = env.get(_PREFIX + "BLACKLIST_APPEND", "")
    if blacklist and legacy:
        raise ValueError(
            "Cannot set both LIGHTX2V_FL_GEMS_BLACKLIST and "
            "LIGHTX2V_FL_GEMS_UNUSED; UNUSED is a legacy alias."
        )
    blacklist = blacklist or legacy
    if whitelist and blacklist:
        raise ValueError(
            "Cannot set LIGHTX2V_FL_GEMS_WHITELIST with "
            "LIGHTX2V_FL_GEMS_BLACKLIST or LIGHTX2V_FL_GEMS_UNUSED."
        )
    if whitelist:
        return _parse_ops(whitelist), None
    if blacklist:
        selected = _parse_ops(blacklist)
    else:
        defaults = _read_defaults()
        selected = list(defaults["platforms"].get(vendor_name, []))
        model = model_name if model_name is not None else env.get(_PREFIX + "MODEL", "")
        model_defaults = defaults["models"].get(model, {})
        selected += model_defaults.get("*", [])
        selected += model_defaults.get(vendor_name, [])
    # Make fresh lists and retain insertion order, including deployment additions.
    selected = list(dict.fromkeys(selected + _parse_ops(append)))
    return None, selected or ([] if blacklist or append else None)
