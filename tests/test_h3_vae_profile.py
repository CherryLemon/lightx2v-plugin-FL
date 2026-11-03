"""Registration ownership, precision boundary and failure restoration."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "lightx2v_fl" / "h3_vae_profile.py"
    spec = importlib.util.spec_from_file_location("h3_profile_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.H3VAEGemsProfile


@pytest.fixture
def runtime(monkeypatch):
    history = []
    gems = SimpleNamespace(current_work_registrar=None)
    class Context:
        def __init__(self, exclude):
            self.exclude = tuple(exclude)
        def __enter__(self):
            assert gems.current_work_registrar is None
            gems.current_work_registrar = self
            history.append(("enter", self.exclude))
        def __exit__(self, *_):
            history.append(("exit", self.exclude))
            gems.current_work_registrar = None
    gems.use_gems = lambda exclude: Context(exclude)
    gems.linear = lambda value, weight, bias: ("gems", value)
    monkeypatch.setitem(sys.modules, "flag_gems", gems)
    class RMSNorm:
        normalized_shape = (64,)
        weight = None
        eps = 1e-5
        def forward(self, value):
            return ("base-rms", value)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None),
        nn=SimpleNamespace(RMSNorm=RMSNorm), is_grad_enabled=lambda: False))
    monkeypatch.setitem(sys.modules, "flag_gems.ops._fused_rms_norm",
                        SimpleNamespace(_fused_rms_norm_forward=lambda *a: (("gems-rms", a[0]), None)))
    class Runner:
        config = {"model_cls": "minimax_h3"}
        def __init__(self):
            projection = SimpleNamespace(
                weight=object(), bias=None, forward=lambda value: ("base", value))
            self.norm = RMSNorm()
            self.video_vae = SimpleNamespace(decoder=SimpleNamespace(
                proj_in=projection, modules=lambda: [self.norm]))
        def run_request(self, fail=False):
            return self.run_vae_decoder(fail)
        def run_vae_decoder(self, fail=False):
            if fail:
                raise ValueError("decode failed")
            return (gems.current_work_registrar.exclude,
                    self.video_vae.decoder.proj_in.forward(123), self.norm.forward(123))
    return gems, history, Runner()


def test_phase_routing_and_restoration(runtime):
    gems, history, runner = runtime
    request, decode = runner.run_request, runner.run_vae_decoder
    projection = runner.video_vae.decoder.proj_in.forward
    with load()(runner, base_exclusions=["mm"]) as profile:
        extra = tuple(profile.vae)
        assert runner.run_request() == (extra, ("gems", 123), ("gems-rms", 123))
        assert runner.norm.forward(123) == ("base-rms", 123)
        assert gems.current_work_registrar.exclude == ("mm",)
        assert runner.video_vae.decoder.proj_in.forward is projection
        with profile.current_only():
            assert runner.run_request() == (("mm",), ("base", 123), ("base-rms", 123))
    assert gems.current_work_registrar is None
    assert runner.run_request == request and runner.run_vae_decoder == decode
    assert history == [("enter", ("mm",)), ("exit", ("mm",)),
                       ("enter", extra), ("exit", extra),
                       ("enter", ("mm",)), ("exit", ("mm",))]


def test_decode_error_restores_before_caller_handles_it(runtime):
    gems, _, runner = runtime
    projection = runner.video_vae.decoder.proj_in.forward
    with load()(runner, base_exclusions=["mm"]):
        with pytest.raises(ValueError, match="decode failed"):
            runner.run_request(True)
        assert gems.current_work_registrar.exclude == ("mm",)
        assert runner.video_vae.decoder.proj_in.forward is projection


def test_synchronization_error_still_restores_registration(runtime, monkeypatch):
    gems, _, runner = runtime
    projection = runner.video_vae.decoder.proj_in.forward
    calls = []
    def synchronize():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("sync failed")
    monkeypatch.setattr(sys.modules["torch"].cuda, "synchronize", synchronize)
    with load()(runner, base_exclusions=["mm"]):
        with pytest.raises(RuntimeError, match="sync failed"):
            runner.run_request()
        assert gems.current_work_registrar.exclude == ("mm",)
        assert runner.video_vae.decoder.proj_in.forward is projection


def test_existing_registration_is_preserved(runtime):
    gems, history, runner = runtime
    original = object()
    gems.current_work_registrar = original
    with pytest.raises(RuntimeError, match="own global"):
        load()(runner, base_exclusions=["mm"]).__enter__()
    assert gems.current_work_registrar is original and history == []


def test_other_model_and_missing_vae_are_rejected(runtime):
    _, _, runner = runtime
    runner.config = {"model_cls": "other"}
    with pytest.raises(TypeError):
        load()(runner, base_exclusions=[])
    runner.config = {"model_cls": "minimax_h3"}
    del runner.video_vae
    with pytest.raises(TypeError):
        load()(runner, base_exclusions=[])


def test_control_requires_active_context(runtime):
    _, _, runner = runtime
    with pytest.raises(RuntimeError, match="active"):
        with load()(runner, base_exclusions=[]).current_only():
            pass


def test_direct_rms_rejects_gradients_and_restores(runtime, monkeypatch):
    gems, _, runner = runtime
    original = runner.norm.forward
    monkeypatch.setattr(sys.modules["torch"], "is_grad_enabled", lambda: True)
    with load()(runner, base_exclusions=["mm"]):
        with pytest.raises(RuntimeError, match="without gradients"):
            runner.run_request()
        assert runner.norm.forward == original
        assert gems.current_work_registrar.exclude == ("mm",)
