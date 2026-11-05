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

"""Explicit quantized Sage attention; never a dense-SDPA fallback alias."""

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.ops.attn.template import AttnWeightTemplate

from lightx2v_fl.dispatch import dispatch_op


@ATTN_WEIGHT_REGISTER("flagos_sage_attn")
class FlagOSSageAttnWeight(AttnWeightTemplate):
    """Dense inference H3 attention, preserving the host's flattened output ABI.

    The host passes block_idx/scheduler as scheduling metadata; this dense
    kernel does not consume them. Unknown numerical options are rejected.
    Single-sequence prefix tensors are accepted as in the host's dense Sage
    weight. They must describe the full sequence; packed variable-length batches
    are unsupported. No CPU read of device prefixes is introduced in the hot path.
    """

    _fl_dispatch_operator = "sage_attention"

    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        causal=False,
        block_idx=None,
        scheduler=None,
        **kwargs,
    ):
        if causal is not False:
            raise NotImplementedError(
                "Sage attention supports noncausal H3 inference only"
            )
        if kwargs:
            raise TypeError(f"Unsupported Sage attention parameters: {sorted(kwargs)}")
        if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
            raise ValueError("Sage attention expects [N,H,D] or [B,N,H,D]")
        q4, k4, v4 = (x.unsqueeze(0) if x.ndim == 3 else x for x in (q, k, v))
        for prefix in (cu_seqlens_q, cu_seqlens_kv):
            if prefix is not None and (
                prefix.ndim != 1 or prefix.numel() != 2 or q4.shape[0] != 1
            ):
                raise NotImplementedError(
                    "Packed variable-length Sage batches are unsupported"
                )
        if max_seqlen_q is not None and max_seqlen_q != q4.shape[1]:
            raise ValueError("max_seqlen_q must describe the full dense query")
        if max_seqlen_kv is not None and max_seqlen_kv != k4.shape[1]:
            raise ValueError("max_seqlen_kv must describe the full dense key")
        out = dispatch_op(
            "sage_attention",
            q4,
            k4,
            v4,
            torch_impl=None,
            tensor_layout="NHD",
            sm_scale=None,
            is_causal=False,
            return_lse=False,
            pv_dtype="bf16",
            implementation="auto",
            config=self.config,
        )
        return out.reshape(-1, q4.shape[-2] * q4.shape[-1])
