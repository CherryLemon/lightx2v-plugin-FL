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

"""H3 inference scope: registration, scheduling and public fusion calls."""

import importlib
import inspect
from collections import ChainMap
from contextvars import ContextVar
from functools import wraps
from threading import RLock

import torch
from loguru import logger

from lightx2v_fl.aten_policy import get_global_gems_lists
from lightx2v_fl.dispatch import dispatch_op

from ._patches import ScopedAttributes, ScopedPatches
from .h3_vae import H3VaeFusions

_PROCESS_LOCK = RLock()
_ACTIVE_MODEL = ContextVar("lightx2v_h3_active_model", default=None)
_OWNER = None


def _model_objects(model):
    visited, pending = set(), [(model, 0)]
    while pending:
        obj, depth = pending.pop()
        if id(obj) in visited or depth > 12:
            continue
        visited.add(id(obj))
        if isinstance(obj, torch.Tensor):
            continue
        yield obj
        if isinstance(obj, dict):
            pending.extend((v, depth + 1) for v in obj.values())
        elif isinstance(obj, (tuple, list)):
            pending.extend((v, depth + 1) for v in obj)
        elif hasattr(obj, "__dict__") and not inspect.ismodule(obj):
            pending.extend((v, depth + 1) for v in vars(obj).values())


def _owned_mm(model, cls):
    return {id(obj) for obj in _model_objects(model) if type(obj) is cls}


class H3InferenceContext:
    """Exclusive process-owned ATen scope for one initialized H3 runner.

    The caller must not concurrently register global ATen overrides or execute an
    unrelated runner in this process. VAE arithmetic is native except for explicitly
    selected fusions. This context does not change model loading/offload/dtypes.
    """

    def __init__(self, runner, *, vae_fusion=True):
        if runner.config.get("model_cls") != "minimax_h3":
            raise TypeError(
                "H3InferenceContext requires an initialized MiniMax-H3 runner"
            )
        for name in ("run_request", "run_vae_decoder"):
            if not callable(getattr(runner, name, None)):
                raise TypeError(f"Missing runner contract: {name}")
        self.runner = runner
        self.vae_fusion = vae_fusion
        self.active = False
        self.registration = None
        self.registration_owner = None
        self.patches = ScopedPatches()
        self.parameters = ScopedAttributes()
        self.fusion = (
            H3VaeFusions(runner.video_vae, runner.config) if vae_fusion else None
        )
        self.infer_module = importlib.import_module(
            "lightx2v.models.networks.minimax_h3.infer.transformer_infer"
        )
        self.mm_class = importlib.import_module(
            "lightx2v.common.ops.mm.mm_weight"
        ).MMWeight
        inspect.signature(self.infer_module.apply_swiglu_linear).bind(None, None, None)
        self.owned_mm = _owned_mm(runner.model, self.mm_class)
        self._prepare_patches()

    def _op(self, name, *args, **kwargs):
        return dispatch_op(
            name, *args, torch_impl=None, config=self.runner.config, **kwargs
        )

    def _prepare_patches(self):
        for obj in _model_objects(self.runner.model):
            if getattr(type(obj), "_fl_dispatch_operator", None) in {
                "sage_attention",
                "h3_rms_norm",
            }:
                # Host attention factories construct these weights without a
                # runner config. Preserve per-weight overrides and inherit the
                # model policy only during this owned inference scope.
                self.parameters.add(
                    obj, "config", ChainMap(obj.config, self.runner.config)
                )
        original_request = self.runner.run_request
        original_decode = self.runner.run_vae_decoder
        original_infer = self.runner.model.infer
        original_swiglu = self.infer_module.apply_swiglu_linear
        original_step = self.runner.scheduler._step

        pre = self.runner.model.pre_infer
        if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
            original_rotary = pre._rotary_embedding

            @wraps(original_rotary)
            def rotary(positions):
                return self._op(
                    "h3_rotary_tables",
                    positions.to(torch.float32).contiguous(),
                    freq_dim=pre.rope_freq_dim,
                    theta=pre.rope_theta,
                )

            self.patches.add(pre, "_rotary_embedding", rotary)

        for owner in (self.runner.video_vae, self.runner.audio_vae):
            old_decode = owner.decode

            @wraps(old_decode)
            def cpu_output(*args, _decode=old_decode, **kwargs):
                if kwargs.get("return_cpu") is None:
                    kwargs["return_cpu"] = True
                return _decode(*args, **kwargs)

            self.patches.add(owner, "decode", cpu_output)

        @wraps(original_request)
        def request(*args, **kwargs):
            with _PROCESS_LOCK:
                return original_request(*args, **kwargs)

        @wraps(original_infer)
        def infer(*args, **kwargs):
            token = _ACTIVE_MODEL.set(self)
            try:
                return original_infer(*args, **kwargs)
            finally:
                _ACTIVE_MODEL.reset(token)

        @wraps(original_decode)
        def decode(*args, **kwargs):
            with _PROCESS_LOCK:
                self._synchronize()
                self._disable_registration()
                try:
                    if self.fusion is None:
                        return original_decode(*args, **kwargs)
                    with self.fusion:
                        return original_decode(*args, **kwargs)
                finally:
                    try:
                        self._synchronize()
                    finally:
                        self._enable_registration()

        @wraps(original_swiglu)
        def swiglu(module, value, gate):
            if _ACTIVE_MODEL.get() is not self or not self._supports_swiglu(
                module, value, gate
            ):
                return original_swiglu(module, value, gate)
            packed = value.as_strided(
                (value.shape[0], 28672),
                (28672, 1),
                storage_offset=value.storage_offset(),
            )
            return module.apply(self._op("h3_swiglu", packed))

        @wraps(original_step)
        def step(sample, prediction, timestep, sigmas, step_index, step_update):
            if step_update != "reference_blend":
                return original_step(
                    sample, prediction, timestep, sigmas, step_index, step_update
                )
            return self._op(
                "h3_reference_blend", sample, prediction, timestep, sigmas, step_index
            )

        for owner, name, replacement in [
            (self.runner, "run_request", request),
            (self.runner, "run_vae_decoder", decode),
            (self.runner.model, "infer", infer),
            (self.infer_module, "apply_swiglu_linear", swiglu),
            (self.runner.scheduler, "_step", step),
        ]:
            self.patches.add(owner, name, replacement)

    def _supports_swiglu(self, module, value, gate):
        if (
            id(module) not in self.owned_mm
            or type(module) is not self.mm_class
            or module.has_lora_branch
            or module.has_diff
        ):
            return False
        if (
            value.ndim != 2
            or value.shape != gate.shape
            or value.shape[0] < 1108
            or value.shape[1] != 14336
        ):
            return False
        if (
            value.dtype != torch.bfloat16
            or gate.dtype != value.dtype
            or value.device.type != "cuda"
            or gate.device != value.device
        ):
            return False
        if value.stride() != (28672, 1) or gate.stride() != value.stride():
            return False
        if (
            value.untyped_storage().data_ptr() != gate.untyped_storage().data_ptr()
            or gate.storage_offset() != value.storage_offset() + 14336
        ):
            return False
        return (
            value.storage_offset() + value.shape[0] * 28672
        ) * value.element_size() <= value.untyped_storage().nbytes()

    def _synchronize(self):
        from lightx2v_fl.device import _detect_torch_device

        getattr(torch, _detect_torch_device()).synchronize()

    def _enable_registration(self):
        import flag_gems

        if getattr(flag_gems, "current_work_registrar", None) is not None:
            raise RuntimeError("Global ATen registration is owned by another caller")
        include, exclude = get_global_gems_lists(
            vendor_name=flag_gems.vendor_name, model_name="minimax_h3"
        )
        if include is not None and not include:
            raise ValueError("An explicit empty whitelist cannot enable all operators")
        context = flag_gems.use_gems(include=include, exclude=exclude)
        context.__enter__()
        self.registration = context
        self.registration_owner = getattr(flag_gems, "current_work_registrar", None)

    def _disable_registration(self):
        if self.registration is None:
            return
        import flag_gems

        if (
            getattr(flag_gems, "current_work_registrar", None)
            is not self.registration_owner
        ):
            raise RuntimeError("Global ATen registration ownership changed")
        context, self.registration = self.registration, None
        context.__exit__(None, None, None)
        self.registration_owner = None

    def __enter__(self):
        global _OWNER
        with _PROCESS_LOCK:
            if self.active or _OWNER is not None:
                raise RuntimeError("An H3 inference context is already active")
            if not self.patches.entries:
                self.owned_mm = _owned_mm(self.runner.model, self.mm_class)
                self.fusion = (
                    H3VaeFusions(self.runner.video_vae, self.runner.config)
                    if self.vae_fusion
                    else None
                )
                self._prepare_patches()
            self._enable_registration()
            try:
                self.parameters.install()
                self.patches.install()
            except BaseException:
                try:
                    self.parameters.restore()
                finally:
                    self._disable_registration()
                raise
            self.active = True
            _OWNER = self
        logger.debug("H3 inference scope installed; execution validation is separate")
        return self

    def __exit__(self, exc_type, exc, traceback):
        global _OWNER
        with _PROCESS_LOCK:
            try:
                self.patches.restore()
            finally:
                try:
                    self.parameters.restore()
                finally:
                    try:
                        self._disable_registration()
                    finally:
                        self.active = False
                        self.patches.entries.clear()
                        self.parameters.entries.clear()
                        if self.fusion is not None:
                            self.fusion.patches.entries.clear()
                        self.fusion = None
                        self.owned_mm.clear()
                        if _OWNER is self:
                            _OWNER = None
