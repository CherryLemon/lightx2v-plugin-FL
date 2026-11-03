"""Small, per-weight policy resolver for explicit operator calls."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

OPERATORS = {
    "rms_norm": frozenset({"eps"}),
    "layer_norm": frozenset({"eps"}),
    "attention": frozenset({"scale", "is_causal", "dropout_p"}),
}
BACKENDS = {"flaggems", "torch"}


@lru_cache(maxsize=1)
def _read_defaults():
    path = Path(__file__).parents[1] / "configs" / "dispatch.json"
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_policy(op_name, *, config=None, vendor_name="unknown"):
    """Merge built-in, platform, model (* then vendor), and user fields.

    Each supplied field replaces that field, including the whole parameters
    mapping. No shared template or caller configuration is modified.
    """
    if op_name not in OPERATORS:
        raise ValueError(f"Unknown explicit operator: {op_name!r}")
    config = {} if config is None else config
    if not isinstance(config, Mapping):
        raise TypeError("Weight config must be a mapping")
    defaults = _read_defaults()
    model = defaults["models"].get(config.get("model_cls", ""), {})
    user = config.get("fl_dispatch", {})
    if not isinstance(user, Mapping):
        raise TypeError("fl_dispatch must map operator names to policies")
    unknown = set(user) - OPERATORS.keys()
    if unknown:
        raise ValueError(f"Unknown fl_dispatch operators: {sorted(unknown)}")
    policy = {"order": ["flaggems", "torch"], "parameters": {}}
    for layer in (
        defaults["platforms"].get(vendor_name, {}),
        model.get("*", {}),
        model.get(vendor_name, {}),
        user,
    ):
        entry = layer.get(op_name, {})
        if not isinstance(entry, Mapping) or set(entry) - {"order", "parameters"}:
            raise ValueError(f"Invalid dispatch policy for {op_name!r}")
        policy.update(entry)
    order, parameters = policy["order"], policy["parameters"]
    if (
        not isinstance(order, (list, tuple))
        or not order
        or any(not isinstance(b, str) or b not in BACKENDS for b in order)
        or len(set(order)) != len(order)
    ):
        raise ValueError(f"Invalid backend order for {op_name!r}: {order!r}")
    if not isinstance(parameters, Mapping) or set(parameters) - OPERATORS[op_name]:
        raise ValueError(f"Unsupported parameters for {op_name!r}: {parameters!r}")
    return tuple(order), dict(parameters)
