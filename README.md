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

## Tests

```bash
pytest -q        # CPU-only smoke tests: wiring + torch-fallback correctness
```

## Status

MVP skeleton. The op `apply()` paths call FlagGems where its API is available and
fall back to torch otherwise; per-chip kernel selection and numerical alignment
(esp. fp8/int8 quant gemm) need validation on real hardware. See repo issues /
roadmap.

### MiniMax-H3 H100 VAE inference profile

The optional `H3VAEGemsProfile` applies the validated H3 exclusions for one
dedicated inference process. Outside the VAE decoder it keeps the H3 base
registration. Inside the decoder it uses native linear/addmm and lower-level
FlashAttention dispatch while retaining FlagGems for `decoder.proj_in`.
The input projection must stay on that path to preserve the tested output.

RMSNorm calls the same pinned FlagGems forward kernel directly, avoiding its
unused Autograd wrapper. The shortcut rejects enabled gradients. The context
serializes this runner's requests, requires ownership of global registration,
and restores methods and registrations on exit or decode failure. It depends
on a private FlagGems forward entry point and must be revalidated on upgrades.

Construct the pipeline with `LIGHTX2V_FL_GLOBAL_GEMS=0`; the context then owns
and actively enables global FlagGems dispatch for every inference request.
Use an H3 config selecting `flagos_flash_attn` and `flagos_rms_norm`, with the
required H3 AdaLN cache available. For example, in the configured H3 environment:

```bash
python examples/minimax_h3_h100_vae_profile.py \
  --model-path /path/to/MiniMax-H3 \
  --config /path/to/minimax_h3_flagos.json \
  --output-dir ./results/h3-vae-profile \
  --paired-control --validate-current
```

The output directory must be new. By default the example warms up each profile
and interleaves three current/candidate measurements when `--paired-control`
is selected. End-to-end timing includes media encoding. Optional validation
adds current/candidate/current boundary snapshots outside measured requests;
compare them on CPU outside any FlagGems context.

Validation so far covers the pinned H100 BF16 T2AV workload only: 29 steps,
124 frames, 512x320, seed 42, model CPU offload, FP16 video-VAE linears and FP32
sensitive normalization. LightX2V `c58c16d6`, FlagGems `e3c77c06`, FlagTree
`0.7.1.dev1`, Torch `2.11.0+cu130`, and Triton `3.6.0` were used.

Three paired measurements reduced the current H3 global-ATen profile from
30.206 to 29.799 seconds (1.35%). Video/audio VAE input and output boundaries
were elementwise identical in the same-process A/B/A check. Three samples
show drift; the difference from an older 31.304-second result is not all
attributable to this change. Cross-process output differences remain
unresolved. Other resolutions and Ulysses multi-GPU comparisons are ongoing;
this profile does not claim validation for them.

## License

Apache-2.0
