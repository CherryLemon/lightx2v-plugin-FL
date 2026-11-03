"""Normalisation ops for FlagOS: RMSNorm + LayerNorm.

RMSNorm uses FlagGems' ``rms_norm`` (exported as a top-level op); LayerNorm uses
FlagGems' ``layer_norm`` when present. Policy selects an existing torch
reference before execution when the requested FlagGems API is unavailable.
"""

from __future__ import annotations

import torch
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER
from lightx2v_platform.ops.norm.norm_template import (
    LayerNormWeightTemplate,
    RMSWeightTemplate,
)

from lightx2v_fl.dispatch import dispatch_op


def _rms_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (x.to(in_dtype)) * weight


def _layer_norm_ref(x, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


@RMS_WEIGHT_REGISTER("flagos_rms_norm")
class FlagOSRmsNormWeight(RMSWeightTemplate):
    def __init__(
        self,
        weight_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(
            weight_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
            **kwargs,
        )

    def apply(self, input_tensor):
        x = input_tensor.contiguous()
        return dispatch_op(
            "rms_norm",
            x,
            self.weight,
            eps=self.eps,
            torch_impl=_rms_ref,
            config=self.config,
        )


@LN_WEIGHT_REGISTER("flagos_layer_norm")
class FlagOSLayerNormWeight(LayerNormWeightTemplate):
    def __init__(
        self,
        weight_name=None,
        bias_name=None,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
            **kwargs,
        )

    def apply(self, input_tensor):
        x = input_tensor.contiguous()
        weight = getattr(self, "weight", None)
        bias = getattr(self, "bias", None)
        return dispatch_op(
            "layer_norm",
            x,
            weight,
            bias,
            eps=self.eps,
            torch_impl=_layer_norm_ref,
            config=self.config,
        )
