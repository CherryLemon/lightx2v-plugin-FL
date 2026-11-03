"""Actual apply() callers, using scoped host API doubles and real CPU tensors.

The host doubles supply registration and template construction only. They do
not simulate FlagGems math or establish accelerator/installed-host compatibility.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def callers(monkeypatch):
    monkeypatch.setenv("LIGHTX2V_FL_AUTO_REGISTER", "0")
    dispatch = importlib.import_module("lightx2v_fl.dispatch")
    monkeypatch.setattr(dispatch.backends, "load_flaggems", lambda: None)

    class Registry(dict):
        def __call__(self, key):
            def register(cls):
                self[key] = cls
                return cls

            return register

    class Template:
        def __init__(self, *args, **kwargs):
            self.eps = args[-1] if args else kwargs.get("eps", 1e-6)
            self.config = {}

    modules = {
        "lightx2v.utils.registry_factory": {
            "RMS_WEIGHT_REGISTER": Registry(),
            "LN_WEIGHT_REGISTER": Registry(),
            "ATTN_WEIGHT_REGISTER": Registry(),
        },
        "lightx2v_platform.ops.norm.norm_template": {
            "LayerNormWeightTemplate": Template,
            "RMSWeightTemplate": Template,
        },
        "lightx2v_platform.ops.attn.template": {"AttnWeightTemplate": Template},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    loaded = []
    for name in ("norm", "attn"):
        path = Path(__file__).parents[1] / "lightx2v_fl" / "ops" / (name + ".py")
        spec = importlib.util.spec_from_file_location("isolated_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return dispatch, *loaded


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 16), (2, 3, 16), (0, 16)])
def test_norm_torch_path_against_independent_reference(callers, dtype, shape):
    _, norm, _ = callers
    x = torch.randn(*shape, 2, dtype=dtype)[..., 0]  # non-contiguous
    w = torch.randn(16, dtype=dtype)
    rms = norm.FlagOSRmsNormWeight("w", eps=1e-5, lora_prefix="test")
    rms.weight = w
    rms.config = {
        "fl_dispatch": {"rms_norm": {"order": ["torch"], "parameters": {"eps": 0.02}}}
    }
    out = rms.apply(x)
    ref = (x.double() / torch.sqrt((x.double() ** 2).mean(-1, keepdim=True) + 0.02)).to(
        dtype
    ) * w
    torch.testing.assert_close(
        out, ref, rtol=0.02 if dtype == torch.bfloat16 else 0.002, atol=0.002
    )
    ln = norm.FlagOSLayerNormWeight("w", "b", eps=1e-5)
    ln.weight, ln.bias = w, torch.randn(16, dtype=dtype)
    torch.testing.assert_close(ln.apply(x), F.layer_norm(x, (16,), w, ln.bias, 1e-5))


def test_real_norm_callers_use_gems_and_preserve_contract(callers, monkeypatch):
    dispatch, norm, _ = callers
    calls = []

    def rms_impl(x, normalized_shape, weight, eps):
        calls.append(("rms", x.is_contiguous(), normalized_shape, weight, eps))
        return x

    def ln_impl(x, normalized_shape, weight, bias, eps):
        calls.append(("ln", x.is_contiguous(), normalized_shape, weight, bias, eps))
        return x

    monkeypatch.setattr(
        dispatch.backends,
        "load_flaggems",
        lambda: SimpleNamespace(rms_norm=rms_impl, layer_norm=ln_impl),
    )
    x, w, b = torch.randn(2, 8, 2)[..., 0], torch.ones(8), torch.zeros(8)
    rms = norm.FlagOSRmsNormWeight("w", eps=0.01)
    rms.weight = w
    ln = norm.FlagOSLayerNormWeight("w", "b", eps=0.02)
    ln.weight, ln.bias = w, b
    assert torch.equal(rms.apply(x), x)
    assert torch.equal(ln.apply(x), x)
    assert (
        calls[0][:3] == ("rms", True, (8,)) and calls[0][3] is w and calls[0][4] == 0.01
    )
    assert (
        calls[1][:3] == ("ln", True, (8,))
        and calls[1][3] is w
        and calls[1][4] is b
        and calls[1][5] == 0.02
    )
    ln.weight = None
    ln.bias = None
    torch.testing.assert_close(ln.apply(x), F.layer_norm(x, (8,), eps=0.02))
    assert len(calls) == 2


def test_attention_caller_policy_parameters_and_layout(callers, monkeypatch):
    dispatch, _, attn = callers
    q, k, v = (torch.randn(5, 2, 8) for _ in range(3))
    calls = []

    def gems(q, k, v, scale=None, is_causal=False):
        calls.append((q.shape, k.shape, scale, is_causal))
        return v

    monkeypatch.setattr(
        dispatch.backends,
        "load_flaggems",
        lambda: SimpleNamespace(scaled_dot_product_attention=gems),
    )
    weight = attn.FlagOSFlashAttnWeight()
    weight.config = {
        "fl_dispatch": {"attention": {"parameters": {"scale": 0.25, "is_causal": True}}}
    }
    assert torch.equal(weight.apply(q, k, v), v.reshape(5, 16))
    assert calls == [(torch.Size([1, 2, 5, 8]), torch.Size([1, 2, 5, 8]), 0.25, True)]
    weight.config["fl_dispatch"]["attention"]["order"] = ["torch"]
    out = weight.apply(q, k, v)
    ref = (
        F.scaled_dot_product_attention(
            q.permute(1, 0, 2),
            k.permute(1, 0, 2),
            v.permute(1, 0, 2),
            scale=0.25,
            is_causal=True,
        )
        .permute(1, 0, 2)
        .reshape(5, 16)
    )
    torch.testing.assert_close(out, ref)
    assert len(calls) == 1


def test_attention_apply_error_does_not_execute_torch(callers, monkeypatch):
    dispatch, _, attn = callers

    def fail(q, k, v):
        raise RuntimeError("attention failed")

    monkeypatch.setattr(
        dispatch.backends,
        "load_flaggems",
        lambda: SimpleNamespace(scaled_dot_product_attention=fail),
    )
    monkeypatch.setattr(
        attn, "_torch_sdp", lambda *a, **kw: pytest.fail("runtime fallback")
    )
    q, k, v = (torch.randn(5, 2, 8) for _ in range(3))
    with pytest.raises(RuntimeError, match="attention failed"):
        attn.FlagOSFlashAttnWeight().apply(q, k, v)


def test_norm_default_missing_gems_keeps_main_formula(callers):
    _, norm, _ = callers
    x, w = torch.randn(3, 16), torch.randn(16)
    rms = norm.FlagOSRmsNormWeight("w", eps=1e-6)
    rms.weight = w
    expected = (
        x.double() / torch.sqrt((x.double() ** 2).mean(-1, keepdim=True) + 1e-6)
    ).float() * w
    torch.testing.assert_close(rms.apply(x), expected)


def test_attention_default_missing_gems_keeps_main_formula(callers):
    _, _, attn = callers
    q, k, v = (torch.randn(8, 3, 16) for _ in range(3))
    expected = (
        F.scaled_dot_product_attention(
            q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        )
        .permute(1, 0, 2)
        .reshape(8, 48)
    )
    torch.testing.assert_close(attn.FlagOSFlashAttnWeight().apply(q, k, v), expected)


def test_layer_norm_runtime_error_does_not_fallback(callers, monkeypatch):
    dispatch, norm, _ = callers
    calls = []

    def fail(x, normalized_shape, weight, bias, eps):
        calls.append("gems")
        raise RuntimeError("layer norm failed")

    monkeypatch.setattr(
        dispatch.backends, "load_flaggems", lambda: SimpleNamespace(layer_norm=fail)
    )
    monkeypatch.setattr(
        norm, "_layer_norm_ref", lambda *a, **kw: pytest.fail("runtime fallback")
    )
    layer = norm.FlagOSLayerNormWeight("w", "b")
    layer.weight, layer.bias = torch.ones(8), torch.zeros(8)
    with pytest.raises(RuntimeError, match="layer norm failed"):
        layer.apply(torch.randn(2, 8))
    assert calls == ["gems"]
