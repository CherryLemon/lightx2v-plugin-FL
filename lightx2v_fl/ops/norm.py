"""Normalisation ops for FlagOS: RMSNorm + LayerNorm.

RMSNorm uses FlagGems' ``rms_norm`` (exported as a top-level op); LayerNorm uses
FlagGems' ``layer_norm`` when present. Both degrade to a torch reference.
"""

from __future__ import annotations

import torch

from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER
from lightx2v_platform.ops.norm.norm_template import LayerNormWeightTemplate, RMSWeightTemplate

try:
    import flag_gems

    _GEMS_RMS = getattr(flag_gems, "rms_norm", None)
    _GEMS_LN = getattr(flag_gems, "layer_norm", None)
except Exception:  # pragma: no cover
    flag_gems = None
    _GEMS_RMS = None
    _GEMS_LN = None


def _rms_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (x.to(in_dtype)) * weight


@RMS_WEIGHT_REGISTER("flagos_rms_norm")
class FlagOSRmsNormWeight(RMSWeightTemplate):
    def __init__(self, weight_name, create_cuda_buffer=False, create_cpu_buffer=False,
                 lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=1e-6, **kwargs):
        super().__init__(weight_name, create_cuda_buffer, create_cpu_buffer,
                         lazy_load, lazy_load_file, is_post_adapter, eps, **kwargs)

    def apply(self, input_tensor):
        x = input_tensor.contiguous()
        if _GEMS_RMS is not None:
            try:
                # FlagGems rms_norm(input, normalized_shape, weight, eps)
                return _GEMS_RMS(x, (x.shape[-1],), self.weight, self.eps)
            except TypeError:
                try:
                    return _GEMS_RMS(x, self.weight, self.eps)
                except Exception:
                    pass
            except Exception:
                pass
        return _rms_ref(x, self.weight, self.eps)


@LN_WEIGHT_REGISTER("flagos_layer_norm")
class FlagOSLayerNormWeight(LayerNormWeightTemplate):
    def __init__(self, weight_name=None, bias_name=None, create_cuda_buffer=False,
                 create_cpu_buffer=False, lazy_load=False, lazy_load_file=None,
                 is_post_adapter=False, eps=1e-6, **kwargs):
        super().__init__(weight_name, bias_name, create_cuda_buffer, create_cpu_buffer,
                         lazy_load, lazy_load_file, is_post_adapter, eps,**kwargs)

    def apply(self, input_tensor):
        x = input_tensor.contiguous()
        normalized_shape = (x.shape[-1],)
        weight = getattr(self, "weight", None)
        bias = getattr(self, "bias", None)
        if _GEMS_LN is not None and weight is not None:
            try:
                return _GEMS_LN(x, normalized_shape, weight, bias, self.eps)
            except Exception:
                pass
        return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, self.eps)
