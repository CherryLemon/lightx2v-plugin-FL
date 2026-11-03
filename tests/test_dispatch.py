"""CPU selection tests with recording backends, without a LightX2V install."""

import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def dispatch(monkeypatch):
    monkeypatch.setenv("LIGHTX2V_FL_AUTO_REGISTER", "0")
    module = importlib.import_module("lightx2v_fl.dispatch")
    monkeypatch.setattr(module.backends, "load_flaggems", lambda: None)
    return module


def test_shipped_default_order(dispatch):
    assert dispatch.resolve_policy("rms_norm") == (("flaggems", "torch"), {})


def test_platform_model_user_precedence_and_no_mutation(dispatch, monkeypatch):
    config = importlib.import_module("lightx2v_fl.dispatch.config")
    defaults = {
        "platforms": {
            "nvidia": {"rms_norm": {"order": ["torch"], "parameters": {"eps": 0.1}}}
        },
        "models": {
            "wan": {
                "*": {"rms_norm": {"order": ["flaggems", "torch"]}},
                "nvidia": {"rms_norm": {"parameters": {"eps": 0.2}}},
            }
        },
    }
    monkeypatch.setattr(config, "_read_defaults", lambda: defaults)
    assert dispatch.resolve_policy("rms_norm", vendor_name="nvidia") == (
        ("torch",),
        {"eps": 0.1},
    )
    assert dispatch.resolve_policy(
        "rms_norm", vendor_name="nvidia", config={"model_cls": "wan"}
    ) == (("flaggems", "torch"), {"eps": 0.2})
    user = {
        "model_cls": "wan",
        "fl_dispatch": {"rms_norm": {"order": ["torch"], "parameters": {}}},
    }
    order, parameters = dispatch.resolve_policy(
        "rms_norm", config=user, vendor_name="nvidia"
    )
    parameters["eps"] = 1
    assert order == ("torch",)
    assert user["fl_dispatch"]["rms_norm"]["parameters"] == {}
    assert defaults["models"]["wan"]["nvidia"]["rms_norm"]["parameters"] == {"eps": 0.2}
    assert dispatch.resolve_policy("attention", vendor_name="other") == (
        ("flaggems", "torch"),
        {},
    )


@pytest.mark.parametrize(
    "entry",
    [
        {"order": []},
        {"order": "torch"},
        {"order": ["fake"]},
        {"order": ["torch", "torch"]},
        {"order": [None]},
        {"parameters": {"unknown": True}},
        {"parameters": None},
        {"unknown": True},
        [],
    ],
)
def test_bad_policy_is_rejected(dispatch, entry):
    with pytest.raises(ValueError):
        dispatch.resolve_policy("rms_norm", config={"fl_dispatch": {"rms_norm": entry}})


def test_unknown_op_is_rejected(dispatch):
    with pytest.raises(ValueError, match="Unknown explicit"):
        dispatch.resolve_policy("typo")
    with pytest.raises(ValueError, match="Unknown fl_dispatch"):
        dispatch.resolve_policy("rms_norm", config={"fl_dispatch": {"typo": {}}})


@pytest.mark.parametrize("legacy", [False, True])
def test_rms_api_shapes_and_parameter_forwarding(dispatch, monkeypatch, legacy):
    calls = []
    x, w = torch.randn(2, 8), torch.ones(8)

    def modern(x, normalized_shape, weight, eps=1e-5):
        calls.append((x, normalized_shape, weight, eps))
        return x

    def old(x, weight, eps=1e-5):
        calls.append((x, (x.shape[-1],), weight, eps))
        return x

    gems = SimpleNamespace(rms_norm=old if legacy else modern)
    monkeypatch.setattr(dispatch.backends, "load_flaggems", lambda: gems)
    result = dispatch.dispatch_op(
        "rms_norm",
        x,
        w,
        eps=1e-6,
        torch_impl=lambda *a, **kw: pytest.fail("unexpected fallback"),
        config={"fl_dispatch": {"rms_norm": {"parameters": {"eps": 0.02}}}},
    )
    assert result is x
    assert len(calls) == 1
    assert calls[0][0] is x and calls[0][2] is w
    assert calls[0][1:] == ((8,), w, 0.02)


def test_selection_fallback_and_explicit_order(dispatch, monkeypatch):
    calls = []

    # An existing API with an incompatible signature must not execute.
    def incompatible(x):
        pytest.fail("incompatible implementation executed")

    monkeypatch.setattr(
        dispatch.backends,
        "load_flaggems",
        lambda: SimpleNamespace(rms_norm=incompatible),
    )
    x, w = torch.randn(2, 8), torch.ones(8)

    def native(x, weight, eps):
        calls.append(eps)
        return x

    assert dispatch.dispatch_op("rms_norm", x, w, eps=1e-6, torch_impl=native) is x
    assert calls == [1e-6]
    with pytest.raises(dispatch.NoImplementationError, match="rms_norm"):
        dispatch.dispatch_op(
            "rms_norm",
            x,
            w,
            eps=1e-6,
            torch_impl=native,
            config={"fl_dispatch": {"rms_norm": {"order": ["flaggems"]}}},
        )
    assert calls == [1e-6]


@pytest.mark.parametrize("error", [RuntimeError, TypeError, NotImplementedError])
def test_execution_failure_never_retries(dispatch, monkeypatch, error):
    counts = {"gems": 0, "torch": 0}

    def broken(x, normalized_shape, weight, eps):
        counts["gems"] += 1
        x.add_(1)  # Observable side effect before the failure.
        raise error("kernel failed")

    def native(*args, **kwargs):
        counts["torch"] += 1

    monkeypatch.setattr(
        dispatch.backends, "load_flaggems", lambda: SimpleNamespace(rms_norm=broken)
    )
    x = torch.zeros(1, 4)
    with pytest.raises(error, match="kernel failed"):
        dispatch.dispatch_op("rms_norm", x, torch.ones(4), eps=1e-6, torch_impl=native)
    assert counts == {"gems": 1, "torch": 0}
    assert torch.equal(x, torch.ones_like(x))


def test_dense_attention_does_not_probe_varlen_or_tuple_api(dispatch, monkeypatch):
    def unsupported(*args, **kwargs):
        pytest.fail("non-SDPA API executed")

    gems = SimpleNamespace(
        ops=SimpleNamespace(
            attention=SimpleNamespace(
                flash_attn_varlen_func=unsupported, flash_attention_forward=unsupported
            )
        )
    )
    monkeypatch.setattr(dispatch.backends, "load_flaggems", lambda: gems)
    sentinel = object()
    assert (
        dispatch.dispatch_op("attention", 1, 2, 3, torch_impl=lambda q, k, v: sentinel)
        is sentinel
    )


def test_attention_signature_fallback_preserves_parameters(dispatch, monkeypatch):
    calls = []

    def no_scale(q, k, v):
        pytest.fail("unsupported parameter silently dropped")

    gems = SimpleNamespace(scaled_dot_product_attention=no_scale)
    monkeypatch.setattr(dispatch.backends, "load_flaggems", lambda: gems)

    def native(q, k, v, scale=None):
        calls.append((q, k, v, scale))
        return v

    result = dispatch.dispatch_op(
        "attention",
        1,
        2,
        3,
        torch_impl=native,
        config={"fl_dispatch": {"attention": {"parameters": {"scale": 0.2}}}},
    )
    assert result == 3 and calls == [(1, 2, 3, 0.2)]


def test_optional_dependency_import_failure_is_cached_before_compute(monkeypatch):
    monkeypatch.setenv("LIGHTX2V_FL_AUTO_REGISTER", "0")
    import sys

    from lightx2v_fl.dispatch.backends import load_flaggems

    load_flaggems.cache_clear()
    monkeypatch.setitem(sys.modules, "flag_gems", None)
    try:
        assert load_flaggems() is None
        assert load_flaggems() is None
        assert load_flaggems.cache_info().misses == 1
    finally:
        load_flaggems.cache_clear()


@pytest.mark.parametrize("config", [[], {"fl_dispatch": []}])
def test_invalid_config_type(dispatch, config):
    with pytest.raises(TypeError):
        dispatch.resolve_policy("attention", config=config)
