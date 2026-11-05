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

"""Instance-only VAE orchestration through public, policy-selected operators."""

from functools import wraps

import torch

from lightx2v_fl.dispatch import dispatch_op

from ._patches import ScopedPatches


class H3VaeFusions:
    def __init__(self, model, config):
        self.patches = ScopedPatches()
        self.config = config
        self.active = False
        blocks = list(model.decoder.transformer_blocks)
        if len(blocks) != 36:
            raise NotImplementedError(
                "H3 VAE fusion requires the36-block decoder contract"
            )
        for block in blocks:
            attention = block.attn
            if (
                attention.dim_head != 64
                or attention.sensitive_layer_dtype != torch.float32
                or attention.norm_q.weight is not None
                or attention.norm_k.weight is not None
                or attention.norm_q.eps != attention.norm_k.eps
                or attention.to_qkv is not None
            ):
                raise NotImplementedError(
                    "Unsupported H3 VAE Q/K normalization or projection contract"
                )
            if (
                block.infer_dtype != torch.float16
                or block.sensitive_layer_dtype != torch.float32
                or block.norm1.weight.shape != (2048,)
                or block.norm2.weight.shape != (2048,)
                or any(
                    x.dtype != torch.float32
                    for x in (
                        block.norm1.weight,
                        block.norm2.weight,
                        block.scale1,
                        block.scale2,
                    )
                )
                or block.ff.net[1].p != 0
            ):
                raise NotImplementedError(
                    "Unsupported H3 VAE precision, width or dropout contract"
                )
            self.patches.add(
                attention, "forward", self._attention(attention, attention.forward)
            )
            self.patches.add(block, "forward", self._block(block, block.forward))
            self.patches.add(
                block.ff.net[0], "forward", self._activation(block.ff.net[0])
            )

    def _op(self, name, *args, **kwargs):
        return dispatch_op(name, *args, torch_impl=None, config=self.config, **kwargs)

    def _attention(self, module, original):
        @wraps(original)
        def forward(hidden, rotary_emb=None):
            if (
                rotary_emb is None
                or hidden.ndim != 3
                or hidden.dtype != torch.float16
                or hidden.device.type != "cuda"
            ):
                return original(hidden, rotary_emb)
            batch, tokens, _ = hidden.shape
            if not isinstance(rotary_emb, (tuple, list)) or len(rotary_emb) != 2:
                return original(hidden, rotary_emb)
            cos, sin = rotary_emb
            if any(
                t.dtype != torch.float32
                or t.ndim != 4
                or t.shape[0] not in (1, batch)
                or t.shape[1:] != (tokens, 1, 48)
                or t.stride(-1) != 1
                or t.device != hidden.device
                for t in (cos, sin)
            ):
                return original(hidden, rotary_emb)
            # No fallback is allowed after the first projection starts.
            q = module.to_q(hidden).unflatten(2, (module.heads, module.dim_head))
            k = module.to_k(hidden).unflatten(2, (module.heads, module.dim_head))
            v = module.to_v(hidden).unflatten(2, (module.heads, module.dim_head))
            q, k = self._op(
                "h3_vae_qk_norm_rope", q, k, cos, sin, eps=module.norm_q.eps
            )
            result = module.calculate.apply(
                q, k, v, max_seqlen_q=tokens, max_seqlen_kv=tokens
            )
            return module.to_out[0](result.view(batch, tokens, module.inner_dim))

        return forward

    def _activation(self, module):
        original = module.forward

        @wraps(original)
        def forward(hidden):
            if hidden.dtype != torch.float16 or hidden.device.type != "cuda":
                return original(hidden)
            return self._op("h3_swiglu", module.proj(hidden))

        return forward

    def _block(self, block, original):
        @wraps(original)
        def forward(hidden, rotary_emb=None):
            if (
                hidden.dtype != torch.float32
                or hidden.device.type != "cuda"
                or not hidden.is_contiguous()
                or hidden.shape[-1] != 2048
            ):
                return original(hidden, rotary_emb)
            normed = self._op(
                "h3_vae_residual_norm", hidden, block.norm1.weight, eps=block.norm1.eps
            )
            update = block.attn(normed, rotary_emb)
            residual, normed = self._op(
                "h3_vae_residual_norm",
                hidden,
                block.norm2.weight,
                eps=block.norm2.eps,
                update=update,
                scale=block.scale1,
            )
            update = block.ff(normed)
            return self._op("h3_vae_residual", residual, update, block.scale2)

        return forward

    def __enter__(self):
        if self.active:
            raise RuntimeError("VAE fusion context is already active")
        self.patches.install()
        self.active = True
        return self

    def __exit__(self, *exc):
        try:
            self.patches.restore()
        finally:
            self.active = False
