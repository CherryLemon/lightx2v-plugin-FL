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

"""Owned, atomic and reversible instance/module patches."""


class ScopedPatches:
    def __init__(self):
        self.entries = []
        self.installed = []

    def add(self, owner, name, replacement):
        original = getattr(owner, name)
        if not callable(original) or not callable(replacement):
            raise TypeError(f"Patch target {name} must be callable")
        had_attribute = name in vars(owner)
        self.entries.append((owner, name, original, replacement, had_attribute))

    def install(self):
        if self.installed:
            raise RuntimeError("Patch group is already active")
        for owner, name, original, _, _ in self.entries:
            if getattr(owner, name) != original:
                raise RuntimeError(f"Patch target changed before installation: {name}")
        try:
            for entry in self.entries:
                owner, name, _, replacement, _ = entry
                setattr(owner, name, replacement)
                self.installed.append(entry)
        except BaseException:
            self.restore()
            raise

    def restore(self):
        changed = []
        for owner, name, original, replacement, had_attribute in reversed(
            self.installed
        ):
            if getattr(owner, name) is not replacement:
                changed.append(name)
                continue
            if had_attribute:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.installed.clear()
        if changed:
            raise RuntimeError(
                f"Patch ownership changed; foreign replacements retained: {changed}"
            )


class ScopedAttributes(ScopedPatches):
    """Identity-owned data bindings, with the same atomic restore rules."""

    def add(self, owner, name, replacement):
        original = getattr(owner, name)
        self.entries.append((owner, name, original, replacement, name in vars(owner)))

    def install(self):
        if self.installed:
            raise RuntimeError("Attribute group is already active")
        for owner, name, original, _, _ in self.entries:
            if getattr(owner, name) is not original:
                raise RuntimeError(f"Attribute changed before installation: {name}")
        try:
            for entry in self.entries:
                owner, name, _, replacement, _ = entry
                setattr(owner, name, replacement)
                self.installed.append(entry)
        except BaseException:
            self.restore()
            raise
