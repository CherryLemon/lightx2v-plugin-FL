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

"""H3 native-order RMS; the weight instance owns its resident staging cache."""

import torch
from lightx2v.utils.registry_factory import RMS_WEIGHT_REGISTER

from lightx2v_fl.dispatch import dispatch_op

from .norm import FlagOSRmsNormWeight


@RMS_WEIGHT_REGISTER("flagos_h3_rms_norm")
class FlagOSH3RmsNormWeight(FlagOSRmsNormWeight):
    _fl_dispatch_operator = "h3_rms_norm"

    def _resident_weight(self, device):
        source = self.weight
        if source.device == device:
            return source
        if source.device.type != "cpu":
            raise ValueError("H3 RMS weight must be on the input device or CPU")
        try:
            version = source._version
        except RuntimeError:
            # Inference tensors have no version counter. Never reuse a copy
            # whose source mutations cannot be detected.
            version = None
        key = (id(source), source.data_ptr(), version, device)
        cached = getattr(self, "_fl_resident_weight", None)
        if version is not None and cached is not None and cached[0] == key:
            return cached[2]
        backend = getattr(torch, device.type)
        capture = getattr(backend, "is_current_stream_capturing", lambda: False)
        if capture():
            raise RuntimeError(
                "Stage H3 RMS weights during warmup before graph capture"
            )
        target = source.to(device=device, copy=True)
        self._fl_resident_weight = (
            (key, source, target) if version is not None else None
        )
        return target

    def clear_resident_weight(self):
        self._fl_resident_weight = None

    def apply(self, input_tensor):
        x = input_tensor.contiguous()
        return dispatch_op(
            "h3_rms_norm",
            x,
            self._resident_weight(x.device),
            eps=self.eps,
            tier="auto",
            torch_impl=None,
            config=self.config,
        )
