# MiniMax-H3 integration

This model-specific integration depends on the independent global ATen policy
and explicit operator dispatch changes in PRs #6 and #7. Those PRs contain no
H3 adaptation. The H3 changes, config and lifecycle are maintained here.

Numerical code is in `flaggems_lightx2v` (H3-specific contracts) and `flag_gems`
(generic SageAttention, metadata and GEMM). The plugin calls public APIs. It does
not load experimental source paths or patch installed numerical dependencies.

## Install and run

Install the device-matched PyTorch and compiler first, then the reviewed/pinned
FlagGems and FlagGems-lightx2v revisions. Install this plugin with `pip install
--no-deps .`. The complete [environment and reproduction guide](https://github.com/CherryLemon/FlagGems-lightx2v/blob/5473a1d/docs/reproduction.md)
lists required image components, cache generation and device validation. The
[optimization design](https://github.com/CherryLemon/FlagGems-lightx2v/blob/5473a1d/docs/optimization.md)
describes numerical contracts and ownership.

`examples/minimax_h3_sm90.json` is a single-H100, fully resident, fixed-conditioning
configuration. Set its AdaLN cache directory to a cache generated with the same
model/config using the upstream `tools/cache_minimax_h3_adaln` scripts. The host
revision used during development is `c58c16d607c49bf6021fecc6238ed177752d7e87`.

```bash
CUDA_VISIBLE_DEVICES=0 LIGHTX2V_FL_DISABLE_FLAGCX=1 \
python examples/minimax_h3.py \
  --model-path /models/MiniMax-H3 --config examples/minimax_h3_sm90.json \
  --conditioning /inputs/conditioning.pt --conditioning-sha256 "$CONDITIONING_SHA256" \
  --prompt "$PROMPT" --output /outputs/h3.mp4
ffmpeg -v error -i /outputs/h3.mp4 -f null -
```

The conditioning dictionary contains BF16 `[tokens,5120]` `prompt_embeds` and
INT64 `text_token_tags`, generated for this exact prompt. Fixtures/model weights
are not distributed. For normal text encoding, omit both conditioning arguments
and set `text_encoder_load_on_init=true`; that path has a different memory and
timing scope and requires its own acceptance.

The context owns one runner in an exclusive process: it temporarily registers
the selected DiT ATen implementations, disables global registration during VAE
decode, installs explicit VAE fusions and restores its own patches on exit. It
does not support concurrent unrelated runners or foreign ATen registrars.
Unsupported VAE metadata takes the existing host path before computation;
errors during computation propagate. Quantized Sage has no dense-SDPA retry.

S5000/other domestic devices require their vendor toolchain and host-framework
adaptation. Use BF16 PV with an explicitly supported baseline/TLE implementation,
choose the matching RMS tier, and initially pass `--no-vae-fusion`. The SM90 JSON
must not be used unchanged: FP8 Sage and native-order FP64 rotary tables are
SM90-specific. Operator-level S5000 evidence does not establish complete H3
quality, multi-card compatibility, or performance on another backend.

H3 multi-card configurations use Ulysses **sequence** parallelism (not tensor
parallelism). Configure the upstream `parallel.seq_p_size`, `seq_p_attn_type`
and collective backend only after bounded communication tests. Keep VAE
parallelism, offload, seed, frames, steps and timing boundaries explicit.

## Review checkpoint and numerical acceptance

The pinned draft operator revisions are in `requirements-h3-operators.txt`. Install
with `python -m pip install --no-deps -r requirements-h3-operators.txt` after the
vendor environment is prepared. These pins preserve review provenance.

A packaged360p request passes the complete historical-native final video/audio
latent comparison bitwise. The integration explicitly binds the runner policy
to host-created Sage/RMS weights during the context, preserving per-weight
overrides and restoring the original config objects on exit. This avoids silently
using the BF16-PV default when the model requested SM90 FP8 PV.

This is one fixed-conditioning request, not a repeated steady-state performance
cohort. The complete resolution/Ulysses matrix has not been rerun after packaging;
do not use historical pre-packaging throughput as acceptance of this head.
