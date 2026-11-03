# lightx2v-plugin-FL

A **pluggable** [LightX2V](https://github.com/ModelTC/LightX2V) backend built on
the [FlagOS](https://github.com/FlagOpen) unified multi-chip stack:

- **[FlagGems](https://github.com/FlagOpen/FlagGems)** — a Triton operator
  library that auto-detects the underlying vendor (NVIDIA / Ascend / Cambricon /
  MetaX / MUSA / Kunlun / Iluvatar / …) and runs *one* set of kernels across all
  of them.
- **[FlagCX](https://github.com/FlagOpen/FlagCX)** — a cross-chip collective
  communication library that plugs into PyTorch as a `flagcx` ProcessGroup
  backend.

## Why a `flagos` meta-platform?

LightX2V's `lightx2v_platform` already abstracts domestic chips, but each chip
lives in its own `ops/<attn|mm|norm|rope>/<chip>/` directory with a hand-written
kernel set. That is N adaptations for N chips.

`flagos` is different: it is **one** `PLATFORM=flagos` backend that covers *every*
chip FlagOS supports, by delegating compute to FlagGems and communication to
FlagCX. Add a chip to FlagOS → it works in LightX2V through this plugin, no new
LightX2V code.

```
            ┌──────────────────────────────────────────────┐
            │            LightX2V (host framework)         │
            │   registry_factory: MM/ATTN/RMS/ROPE keys    │
            └────────────────┬─────────────────────────────┘
                             │ "flagos*" registry keys
            ┌────────────────▼─────────────────────────────┐
            │          lightx2v-plugin-FL (this repo)      │
            │  device/  ops/{attn,mm,norm,rope}  enable.py │
            └──────────┬────────────────────────┬──────────┘
               compute │                   comm │
            ┌──────────▼─────────┐   ┌──────────▼──────────┐
            │      FlagGems      │   │        FlagCX       │
            │  (Triton kernels)  │   │  (ProcessGroup PG)  │
            └────────┬───────────┘   └──────────┬──────────┘
   NVIDIA · Ascend · Cambricon · MetaX · MUSA · Kunlun · Iluvatar · …
```

## Install

```bash
# 1. the plugin (this repo)
pip install -e .

# 2. FlagGems for your chip (example: NVIDIA)
pip install "flag_gems[nvidia]"      # or [ascend], [cambricon], ...

# 3. (optional, multi-card) FlagCX torch plugin for your chip
#    see https://github.com/FlagOpen/FlagCX
```

`flag_gems` and `flagcx` are **not** hard dependencies — the plugin degrades to
torch reference implementations when they are absent, so `import` never breaks.

## Use

```bash
PLATFORM=flagos python lightx2v/infer.py \
    --model_cls wan2.1 \
    --model_path /path/to/wan \
    --config_json lightx2v_fl/configs/wan_t2v_flagos.json
```

The config selects the FlagOS ops by registry key:

| Config field        | Value                  |
|---------------------|------------------------|
| `self_attn_1_type`  | `flagos_flash_attn`    |
| `cross_attn_*_type` | `flagos_flash_attn`    |
| `rms_norm_type`     | `flagos_rms_norm`      |
| `layer_norm_type`   | `flagos_layer_norm`    |
| `rope_type`         | `flagos_rope`          |
| `dit_quant_scheme`  | `flagos` / `flagos-fp8` / `flagos-int8` |

### Optional: global FlagGems aten patching

Lower *generic* torch ops (softmax, elementwise, …) to FlagGems too:

```bash
LIGHTX2V_FL_GLOBAL_GEMS=1 PLATFORM=flagos python lightx2v/infer.py ...
```

### Knobs

| Env var | Default | Effect |
|---|---|---|
| `LIGHTX2V_FL_AUTO_REGISTER` | `1` | Register on `import lightx2v_fl`. |
| `LIGHTX2V_FL_GLOBAL_GEMS` | `0` | `flag_gems.enable()` at aten level. |
| `LIGHTX2V_FL_GEMS_UNUSED` | – | Comma-list of aten ops to exclude from global patch. |
| `LIGHTX2V_FL_DISABLE_FLAGCX` | `0` | Use vendor-native CCL instead of FlagCX. |

## Activation: how the plugin is discovered

There are two paths; pick based on whether the upstream hook has landed.

### Preferred — entry point (needs the upstream hook)

`pyproject.toml` exposes:

```toml
[project.entry-points."lightx2v.platform_plugins"]
flagos = "lightx2v_fl:register"
```

Once LightX2V scans this group inside `set_ai_device()` (see
[`docs/upstream-entrypoint-hook.md`](docs/upstream-entrypoint-hook.md)),
`pip install lightx2v-plugin-fl` is all that's required.

### Fallback — import order (works today, zero upstream change)

`import lightx2v_fl` **before** `import lightx2v`. The plugin registers its ops
into the *final* `lightx2v.utils.registry_factory` tables, so it is immune to the
one-shot `merge()` snapshot that copies `PLATFORM_*` registries at framework
import time.

> **The snapshot gotcha.** `registry_factory.py` runs
> `ATTN_WEIGHT_REGISTER.merge(PLATFORM_ATTN_WEIGHT_REGISTER)` at import. `merge`
> is a copy, not a live view — anything registered into `PLATFORM_*` *after* that
> line is invisible. This plugin sidesteps it by registering into the final
> tables directly. The upstream hook fixes it properly by running plugin
> registration *before* the merge.

### Explicit operator dispatch

RMSNorm, LayerNorm and dense attention calls now share a small dispatch entry
point, `lightx2v_fl.dispatch.dispatch_op`. Their existing registry keys, tensor
layouts and default order (`flaggems`, then `torch`) are preserved. Global ATen
registration is separate: its lists do not control these explicit calls. A
`torch` backend can still encounter globally patched ATen ops if you enabled them.
RoPE and the linear/quantized-linear paths retain their current implementations.

Use the weight template's normal `set_config(config)` / `config` entry point:

```python
weight.set_config({
    "model_cls": "wan2.1",
    "fl_dispatch": {
        "rms_norm": {"order": ["torch"]},
        "attention": {
            "order": ["flaggems", "torch"],
            "parameters": {"scale": 0.125, "is_causal": False}
        }
    }
})
```

Policies are merged by field in this order: built-in defaults, detected FlagGems
vendor, model `*` defaults, model/vendor defaults, user `fl_dispatch`. Each supplied
`order` or `parameters` field replaces the entire earlier field; `{}` parameters
clear inherited overrides. Parameters override the caller's defaults (for example
the norm template's `eps`) and are forwarded unchanged to the selected backend.
The shipped [`dispatch.json`](lightx2v_fl/configs/dispatch.json) has empty
`platforms`/`models` maps, so successful default calls retain their behavior.
Maintainers can add policies using this schema:

```json
{
  "platforms": {"nvidia": {"rms_norm": {"order": ["flaggems", "torch"]}}},
  "models": {"wan2.1": {"*": {"attention": {"parameters": {"scale": 0.125}}}}}
}
```

Model keys match `config["model_cls"]` exactly; vendor names come from FlagGems.
Unknown models/vendors add no defaults. Supported operators are `rms_norm`,
`layer_norm`, and `attention`; backends are `flaggems` and `torch`. Norms accept
policy parameter `eps`; attention accepts `scale`, `is_causal`, and `dropout_p`
when the selected API supports them. Empty orders, duplicate/unknown
backends, unknown fields/operators and unsupported parameter names are errors.

Fallback happens **before execution** when an API is absent, its signature cannot
be inspected/bound, or a known contract is unavailable (non-affine LayerNorm).
Use an order containing only `flaggems` to require it; no supported implementation
raises `NoImplementationError`. After a call starts, every error propagates,
including `TypeError` and `NotImplementedError`; failed computations are never
retried on another backend. This deliberately replaces the previous broad
runtime-exception fallback. Legacy RMS signatures are recognized by signature
binding, without trial execution. Dense attention uses only SDPA-compatible APIs;
varlen-only and tuple-returning flash forward APIs are not substituted for SDPA.
Successful calls keep their old layout adaptation and reference math. Hardware
shape/dtype support and numerical behavior still require validation for the
installed vendor build.

Dispatch reads each weight's current configuration, so one weight's override
does not mutate another's policy.

## Tests

```bash
pytest -q tests/test_dispatch.py tests/test_dispatch_callers.py  # isolated CPU contracts
pytest -q tests/test_smoke.py  # requires the LightX2V host; CPU fallback smoke
```

## Status

MVP skeleton. The op `apply()` paths call FlagGems where its API is available and
fall back to torch otherwise; per-chip kernel selection and numerical alignment
(esp. fp8/int8 quant gemm) need validation on real hardware. See repo issues /
roadmap.

## License

Apache-2.0
