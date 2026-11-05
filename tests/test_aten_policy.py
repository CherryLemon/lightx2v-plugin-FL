"""CPU policy and registration contracts; no accelerator kernels are executed."""

import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setenv("LIGHTX2V_FL_AUTO_REGISTER", "0")
    module = importlib.import_module("lightx2v_fl.aten_policy")
    monkeypatch.setattr(
        module,
        "_read_defaults",
        lambda: {
            "platforms": {"nvidia": ["mm", "add"]},
            "models": {"wan": {"*": ["add", "linear"], "nvidia": ["bmm"]}},
        },
    )
    return module


@pytest.mark.parametrize(
    "values, expected",
    [
        ({}, (None, ["mm", "add", "linear", "bmm"])),
        ({"BLACKLIST": "softmax"}, (None, ["softmax"])),
        ({"UNUSED": "softmax"}, (None, ["softmax"])),
        (
            {"BLACKLIST_APPEND": " add, copy_,copy_ "},
            (None, ["mm", "add", "linear", "bmm", "copy_"]),
        ),
        (
            {"BLACKLIST": "linear", "BLACKLIST_APPEND": "add,linear"},
            (None, ["linear", "add"]),
        ),
        ({"UNUSED": "linear", "BLACKLIST_APPEND": "add"}, (None, ["linear", "add"])),
        (
            {"WHITELIST": " rms_norm, softmax ", "BLACKLIST_APPEND": "softmax"},
            (["rms_norm", "softmax"], None),
        ),
        ({"WHITELIST": "", "BLACKLIST": ""}, (None, ["mm", "add", "linear", "bmm"])),
        ({"BLACKLIST": " , "}, (None, [])),
        ({"WHITELIST": " , "}, ([], None)),
        ({"UNUSED": ""}, (None, ["mm", "add", "linear", "bmm"])),
    ],
)
def test_precedence(policy, values, expected):
    env = {"LIGHTX2V_FL_GEMS_" + k: v for k, v in values.items()}
    assert (
        policy.get_global_gems_lists(
            vendor_name="nvidia", model_name="wan", environ=env
        )
        == expected
    )


@pytest.mark.parametrize(
    "values",
    [
        {"WHITELIST": "a", "BLACKLIST": "b"},
        {"WHITELIST": "a", "UNUSED": "b"},
        {"BLACKLIST": "a", "UNUSED": "a"},
        {"WHITELIST": " ", "BLACKLIST": "b"},
    ],
)
def test_conflicts(policy, values):
    with pytest.raises(ValueError, match="Cannot set"):
        policy.get_global_gems_lists(
            environ={"LIGHTX2V_FL_GEMS_" + k: v for k, v in values.items()}
        )


def test_defaults_are_isolated_and_model_is_explicit(policy):
    first = policy.get_global_gems_lists(vendor_name="nvidia", environ={})[1]
    first.append("pollution")
    assert policy.get_global_gems_lists(vendor_name="nvidia", environ={}) == (
        None,
        ["mm", "add"],
    )
    assert policy.get_global_gems_lists(environ={"LIGHTX2V_FL_GEMS_MODEL": "wan"}) == (
        None,
        ["add", "linear"],
    )
    assert policy.get_global_gems_lists(
        model_name="other", environ={"LIGHTX2V_FL_GEMS_MODEL": "wan"}
    ) == (None, None)


def test_shipped_defaults_preserve_main_behavior(monkeypatch):
    monkeypatch.setenv("LIGHTX2V_FL_AUTO_REGISTER", "0")
    from lightx2v_fl.aten_policy import get_global_gems_lists

    assert get_global_gems_lists(
        vendor_name="nvidia", model_name="wan", environ={}
    ) == (None, None)


@pytest.fixture
def enable(monkeypatch):
    monkeypatch.setenv("LIGHTX2V_FL_AUTO_REGISTER", "0")
    for name in list(__import__("os").environ):
        if name.startswith("LIGHTX2V_FL_GEMS_"):
            monkeypatch.delenv(name)
    module = importlib.import_module("lightx2v_fl.enable")
    monkeypatch.setattr(module, "_ENABLED", False)
    return module


@pytest.mark.parametrize(
    "values, expected",
    [
        ({}, ("enable", {"unused": None})),
        ({"UNUSED": " add,mm "}, ("enable", {"unused": ["add", "mm"]})),
        ({"WHITELIST": "softmax"}, ("only_enable", {"include": ["softmax"]})),
        ({"BLACKLIST": " , "}, ("enable", {"unused": []})),
        ({"WHITELIST": " , "}, ("only_enable", {"include": []})),
    ],
)
def test_enable_calls_and_idempotence(enable, monkeypatch, values, expected):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "flag_gems",
        SimpleNamespace(
            vendor_name="nvidia",
            enable=lambda **kw: calls.append(("enable", kw)),
            only_enable=lambda **kw: calls.append(("only_enable", kw)),
        ),
    )
    for k, v in values.items():
        monkeypatch.setenv("LIGHTX2V_FL_GEMS_" + k, v)
    enable.enable_global_flaggems()
    enable.enable_global_flaggems()
    assert calls == [expected]


def test_registration_failure_is_not_cached(enable, monkeypatch):
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("registration failed")

    gems = SimpleNamespace(enable=fail)
    monkeypatch.setitem(sys.modules, "flag_gems", gems)
    with pytest.raises(RuntimeError, match="registration failed"):
        enable.enable_global_flaggems()
    assert not enable._ENABLED
    gems.enable = lambda **kw: calls.append(kw)
    enable.enable_global_flaggems()
    assert calls == [{"unused": None}, {"unused": None}]


def test_missing_whitelist_api_never_enables_everything(enable, monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, "flag_gems", SimpleNamespace(enable=lambda **kw: calls.append(kw))
    )
    monkeypatch.setenv("LIGHTX2V_FL_GEMS_WHITELIST", "softmax")
    with pytest.raises(RuntimeError, match="only_enable"):
        enable.enable_global_flaggems()
    assert calls == []
    assert not enable._ENABLED


def test_missing_dependency_remains_optional(enable, monkeypatch):
    monkeypatch.setitem(sys.modules, "flag_gems", None)
    enable.enable_global_flaggems()
    assert not enable._ENABLED


def test_config_conflict_has_no_registration_side_effect(enable, monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, "flag_gems", SimpleNamespace(enable=lambda **kw: calls.append(kw))
    )
    monkeypatch.setenv("LIGHTX2V_FL_GEMS_WHITELIST", "softmax")
    monkeypatch.setenv("LIGHTX2V_FL_GEMS_BLACKLIST", "mm")
    with pytest.raises(ValueError):
        enable.enable_global_flaggems()
    assert calls == []


@pytest.mark.parametrize(
    "explicit_model, expected", [(None, ["mm", "linear"]), ("other", ["mm"])]
)
def test_enable_uses_detected_vendor_and_selected_model(
    enable, monkeypatch, explicit_model, expected
):
    from lightx2v_fl import aten_policy

    monkeypatch.setattr(
        aten_policy,
        "_read_defaults",
        lambda: {
            "platforms": {"nvidia": ["mm"]},
            "models": {"wan": {"nvidia": ["linear"]}},
        },
    )
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "flag_gems",
        SimpleNamespace(vendor_name="nvidia", enable=lambda **kw: calls.append(kw)),
    )
    monkeypatch.setenv("LIGHTX2V_FL_GEMS_MODEL", "wan")
    enable.enable_global_flaggems(model_name=explicit_model)
    assert calls == [{"unused": expected}]
