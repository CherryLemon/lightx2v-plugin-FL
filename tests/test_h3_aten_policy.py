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

"""Packaged H3 defaults obey the same public policy precedence."""

from lightx2v_fl.aten_policy import get_global_gems_lists


def test_h3_nvidia_policy_is_scoped_and_fresh():
    include, excluded = get_global_gems_lists(
        vendor_name="nvidia", model_name="minimax_h3", environ={}
    )
    assert include is None
    assert excluded == [
        "mm",
        "mm_out",
        "rms_norm",
        "conv1d",
        "conv_transpose1d",
        "index_select",
        "mul",
        "add",
        "neg",
        "cat",
        "silu",
        "to_copy",
        "copy_",
    ]
    excluded.append("caller_mutation")
    assert (
        "caller_mutation"
        not in get_global_gems_lists(
            vendor_name="nvidia", model_name="minimax_h3", environ={}
        )[1]
    )
    assert get_global_gems_lists(
        vendor_name="mthreads", model_name="minimax_h3", environ={}
    ) == (None, None)
    assert get_global_gems_lists(
        vendor_name="nvidia", model_name="unrelated", environ={}
    ) == (None, None)


def test_user_modes_still_override_h3_defaults():
    p = "LIGHTX2V_FL_GEMS_"
    assert get_global_gems_lists(
        vendor_name="nvidia", model_name="minimax_h3", environ={p + "WHITELIST": "mm"}
    ) == (["mm"], None)
    assert get_global_gems_lists(
        vendor_name="nvidia",
        model_name="minimax_h3",
        environ={p + "BLACKLIST": "add", p + "BLACKLIST_APPEND": "add,mm"},
    ) == (None, ["add", "mm"])
    assert get_global_gems_lists(
        vendor_name="nvidia", model_name="minimax_h3", environ={p + "BLACKLIST": " , "}
    ) == (None, [])
