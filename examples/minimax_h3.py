# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""H3 public integration. Fixed conditioning is optional and never bundled."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditioning")
    parser.add_argument("--conditioning-sha256")
    parser.add_argument("--no-vae-fusion", action="store_true")
    args = parser.parse_args()
    if bool(args.conditioning) != bool(args.conditioning_sha256):
        parser.error("Fixed conditioning requires both a file and its expected SHA256")
    os.environ.setdefault("PLATFORM", "flagos")
    os.environ["LIGHTX2V_FL_GLOBAL_GEMS"] = "0"
    import numpy as np
    import torch
    from lightx2v import LightX2VPipeline

    import lightx2v_fl  # noqa: F401 - register before host registry snapshots
    from lightx2v_fl.models.minimax_h3 import H3InferenceContext

    config = json.loads(Path(args.config).read_text())
    if (
        config.get("attn_type") != "flagos_sage_attn"
        or config.get("rms_type") != "flagos_h3_rms_norm"
    ):
        raise ValueError("Use the explicit H3 Sage and native-order RMS registry keys")
    fixed = None
    if args.conditioning:
        fixture = Path(args.conditioning)
        if hashlib.sha256(fixture.read_bytes()).hexdigest() != args.conditioning_sha256:
            raise ValueError("Conditioning file SHA256 mismatch")
        fixed = torch.load(fixture, map_location="cpu", weights_only=True)
        required = {"prompt_embeds", "text_token_tags"}
        if (
            not isinstance(fixed, dict)
            or not required <= set(fixed)
            or set(fixed) - required - {"token_ids"}
        ):
            raise ValueError(
                "Conditioning requires embeddings/tags; optional token_ids are audit metadata"
            )
        if (
            fixed["prompt_embeds"].ndim != 2
            or fixed["prompt_embeds"].shape[-1] != 5120
            or fixed["prompt_embeds"].dtype != torch.bfloat16
        ):
            raise ValueError("Expected two-dimensional BF16 prompt embeddings")
        if (
            fixed["text_token_tags"].dtype != torch.int64
            or fixed["text_token_tags"].numel() != fixed["prompt_embeds"].shape[0]
        ):
            raise ValueError(
                "Token tags must be INT64 and match the embedding token count"
            )
        if config.get("text_encoder_load_on_init", True):
            raise ValueError(
                "Fixed-conditioning config must disable text_encoder_load_on_init"
            )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pipe = LightX2VPipeline(
        task="t2av",
        model_path=args.model_path,
        model_cls="minimax_h3",
        model_variant="fl2av",
    )
    pipe.create_generator(config_json=args.config)
    original_conditioner = pipe.runner.run_text_encoder
    if fixed is not None:
        from lightx2v_fl.device import _detect_torch_device

        backend = getattr(torch, _detect_torch_device())
        device = torch.device(_detect_torch_device(), backend.current_device())
        prepared = {name: fixed[name].to(device, copy=True) for name in required}

        def fixed_conditioner(input_info, keyframes=None, references=None):
            if (
                input_info.task != "t2av"
                or input_info.prompt != args.prompt
                or keyframes
                or references is not None
            ):
                raise ValueError(
                    "Fixed conditioning is valid only for the specified text-to-audio/video request"
                )
            return prepared

        pipe.runner.run_text_encoder = fixed_conditioner
    try:
        with H3InferenceContext(pipe.runner, vae_fusion=not args.no_vae_fusion):
            pipe.generate(
                task="t2av",
                seed=args.seed,
                prompt=args.prompt,
                size=tuple(config["size"]),
                num_frames=config["num_frames"],
                save_result_path=args.output,
            )
    finally:
        pipe.runner.run_text_encoder = original_conditioner


if __name__ == "__main__":
    main()
