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

"""CPU lifecycle/ABI checks. Host doubles do not establish GPU correctness."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytest_plugins = ["test_dispatch_callers"]


def load_op(name, monkeypatch):
    root = Path(__file__).parents[1] / "lightx2v_fl" / "ops"
    package = ModuleType("h3_test_ops")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "h3_test_ops", package)
    spec = importlib.util.spec_from_file_location(
        "h3_test_ops." + name, root / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sage_dense_abi_and_explicit_precision(callers, monkeypatch):
    dispatch, _, _ = callers
    calls = []

    def sage(q, k, v, **kwargs):
        calls.append((q.shape, kwargs))
        return v

    monkeypatch.setattr(
        dispatch.backends, "load_flaggems", lambda: SimpleNamespace(sage_attention=sage)
    )
    weight = load_op("sage_attn", monkeypatch).FlagOSSageAttnWeight()
    weight.config = {
        "fl_dispatch": {
            "sage_attention": {
                "parameters": {"pv_dtype": "fp8", "implementation": "sm90"}
            }
        }
    }
    q = torch.randn(17, 2, 128)
    assert torch.equal(
        weight.apply(
            q, q, q, cu_seqlens_q=torch.tensor([0, 17]), max_seqlen_q=17, causal=False
        ),
        q.reshape(17, 256),
    )
    assert calls[0][0] == (1, 17, 2, 128)
    assert calls[0][1]["pv_dtype"] == "fp8"
    with pytest.raises(NotImplementedError, match="variable-length"):
        weight.apply(q, q, q, cu_seqlens_q=torch.tensor([0, 8, 17]))
    assert len(calls) == 1
    with pytest.raises(NotImplementedError, match="noncausal"):
        weight.apply(q, q, q, causal=True)
    assert len(calls) == 1


def test_specialized_execution_error_is_not_retried(callers, monkeypatch):
    dispatch, _, _ = callers
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(
        dispatch.backends,
        "load_lightx2v_gems",
        lambda: SimpleNamespace(h3_swiglu=broken),
    )
    with pytest.raises(RuntimeError, match="kernel failure"):
        dispatch.dispatch_op(
            "h3_swiglu", torch.empty(1, 4), torch_impl=lambda *a: pytest.fail("retry")
        )
    assert calls == [1]


def test_h3_rms_same_device_has_no_staging(callers, monkeypatch):
    cls = load_op("h3_norm", monkeypatch).FlagOSH3RmsNormWeight
    obj = cls("weight")
    with torch.inference_mode():
        obj.weight = torch.ones(128)
    assert obj._resident_weight(torch.device("cpu")) is obj.weight
    obj.clear_resident_weight()
    assert obj._fl_resident_weight is None


def test_patch_group_prevalidation_and_owner_cleanup():
    from lightx2v_fl.models._patches import ScopedPatches

    old, replacement, foreign = lambda: 0, lambda: 1, lambda: 2
    first, second = SimpleNamespace(fn=old), SimpleNamespace(fn=old)
    scope = ScopedPatches()
    scope.add(first, "fn", replacement)
    scope.add(second, "fn", replacement)
    second.fn = foreign
    with pytest.raises(RuntimeError, match="changed before"):
        scope.install()
    assert first.fn is old
    second.fn = old
    scope.install()
    second.fn = foreign
    with pytest.raises(RuntimeError, match="foreign replacements"):
        scope.restore()
    assert first.fn is old and second.fn is foreign


def test_h3_context_registration_and_failure_cleanup(monkeypatch):
    monkeypatch.setenv("LIGHTX2V_FL_AUTO_REGISTER", "0")
    module = importlib.import_module("lightx2v_fl.models.minimax_h3")
    mm_module = ModuleType("lightx2v.common.ops.mm.mm_weight")
    mm_module.MMWeight = type("MMWeight", (), {})
    infer_module = ModuleType(
        "lightx2v.models.networks.minimax_h3.infer.transformer_infer"
    )
    infer_module.apply_swiglu_linear = lambda module, value, gate: "original"
    monkeypatch.setitem(sys.modules, mm_module.__name__, mm_module)
    monkeypatch.setitem(sys.modules, infer_module.__name__, infer_module)
    gems = ModuleType("flag_gems")
    gems.current_work_registrar = None
    gems.vendor_name = "nvidia"
    events = []

    class Registration:
        def __enter__(self):
            assert gems.current_work_registrar is None
            gems.current_work_registrar = self
            events.append("enable")

        def __exit__(self, *exc):
            assert gems.current_work_registrar is self
            gems.current_work_registrar = None
            events.append("disable")

    gems.use_gems = lambda **kw: Registration()
    monkeypatch.setitem(sys.modules, "flag_gems", gems)
    decode_calls = []

    def decode(*args, **kwargs):
        decode_calls.append(kwargs["return_cpu"])

    class AttentionWeight:
        _fl_dispatch_operator = "sage_attention"

        def __init__(self):
            self.config = {}

    attention = AttentionWeight()
    original_config = attention.config
    model = SimpleNamespace(
        infer=lambda: None, pre_infer=SimpleNamespace(), attention=attention
    )
    runner = SimpleNamespace(
        config={
            "model_cls": "minimax_h3",
            "fl_dispatch": {"sage_attention": {"parameters": {"pv_dtype": "fp8"}}},
        },
        model=model,
        video_vae=SimpleNamespace(decode=decode),
        audio_vae=SimpleNamespace(decode=decode),
        scheduler=SimpleNamespace(_step=lambda *a: "native"),
        run_request=lambda: None,
        run_vae_decoder=lambda: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )
    originals = (
        runner.run_request,
        runner.run_vae_decoder,
        model.infer,
        infer_module.apply_swiglu_linear,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ctx = module.H3InferenceContext(runner, vae_fusion=False)
    monkeypatch.setattr(ctx, "_synchronize", lambda: events.append("sync"))
    for _ in range(2):
        with ctx:
            assert (
                attention.config["fl_dispatch"]["sage_attention"]["parameters"][
                    "pv_dtype"
                ]
                == "fp8"
            )
            assert attention.config.maps[0] is original_config
            runner.video_vae.decode()
            runner.audio_vae.decode(return_cpu=False)
            with pytest.raises(RuntimeError, match="decode failed"):
                runner.run_vae_decoder()
            assert gems.current_work_registrar is ctx.registration_owner
            assert runner.scheduler._step(1, 2, 3, 4, 5, "other") == "native"
            with pytest.raises(RuntimeError, match="already active"):
                ctx.__enter__()
        assert gems.current_work_registrar is None
        assert attention.config is original_config
        assert (
            runner.run_request,
            runner.run_vae_decoder,
            model.infer,
            infer_module.apply_swiglu_linear,
        ) == originals
        assert not ctx.patches.entries and not ctx.owned_mm
    assert decode_calls == [True, False, True, False]
    assert events.count("enable") == events.count("disable") == 4


def test_scoped_data_binding_preserves_foreign_owner():
    from lightx2v_fl.models._patches import ScopedAttributes

    original, replacement, foreign = {}, {"tier": "sm90"}, {"tier": "sm90"}
    owner = SimpleNamespace(config=original)
    scope = ScopedAttributes()
    scope.add(owner, "config", replacement)
    scope.install()
    owner.config = foreign
    with pytest.raises(RuntimeError, match="foreign replacements"):
        scope.restore()
    assert owner.config is foreign
