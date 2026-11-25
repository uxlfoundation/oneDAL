#!/usr/bin/env python3
# ===============================================================================
# Copyright Contributors to the oneDAL Project
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
# ===============================================================================
"""Re-pins the conda-forge packages of dev/bazel/deps/gcc.bzl.

Renovate bumps `_GCC_VERSION` there, but cannot compute what a conda package
pin also needs: the build string and the sha256 of the `.conda` file. This
script fills both in for every `_pkg(...)` entry, taking the highest build
number conda-forge has published for the entry's version and subdir. Every
entry is re-pinned, including the build numbers of an unchanged version.

    python3 dev/bazel/deps/relock_conda.py          # rewrite gcc.bzl
    python3 dev/bazel/deps/relock_conda.py --check  # exit 1 if it would change
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

GCC_BZL = Path(__file__).with_name("gcc.bzl")

# _pkg("name", <version>, "build", "sha256"[, subdir = "noarch"]),
# where <version> is a string literal or a module-level constant.
PKG = re.compile(
    r'_pkg\("(?P<name>[^"]+)",\s*(?P<version>"[^"]+"|\w+),\s*'
    r'"(?P<build>[^"]+)",\s*"(?P<sha256>[0-9a-f]{64})"'
    r'(?:,\s*subdir\s*=\s*"(?P<subdir>[^"]+)")?\)'
)


def latest_build(name, version, subdir, current_build):
    url = f"https://api.anaconda.org/package/conda-forge/{name}/files"
    # The URL is a fixed https endpoint; `name` comes from gcc.bzl, not user input.
    with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
        files = json.load(response)
    matching = [
        f
        for f in files
        if f["version"] == version
        and f["attrs"]["subdir"] == subdir
        and f["basename"].endswith(".conda")
    ]
    if not matching:
        sys.exit(f"conda-forge has no {subdir} .conda file of {name} {version}")
    top = max(f["attrs"]["build_number"] for f in matching)
    # Variants of one build number differ only in their hash string; keep the
    # pinned one rather than flip between them.
    best = sorted(
        (f for f in matching if f["attrs"]["build_number"] == top),
        key=lambda f: (f["attrs"]["build"] != current_build, f["attrs"]["build"]),
    )[0]
    return best["attrs"]["build"], best["sha256"]


def relock(text):
    constants = dict(re.findall(r'^(\w+)\s*=\s*"([^"]*)"', text, re.MULTILINE))

    def repin(match):
        version = match["version"]
        version = version.strip('"') if version.startswith('"') else constants[version]
        build, sha256 = latest_build(
            match["name"], version, match["subdir"] or "linux-64", match["build"]
        )
        old = match.group(0)
        return old.replace(match["build"], build, 1).replace(match["sha256"], sha256, 1)

    return PKG.sub(repin, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report whether gcc.bzl is stale")
    args = parser.parse_args()
    text = GCC_BZL.read_text()
    updated = relock(text)
    if updated == text:
        print(f"{GCC_BZL.name}: up to date")
        return 0
    if args.check:
        print(f"{GCC_BZL.name}: stale; run {Path(__file__).name} to re-pin it")
        return 1
    GCC_BZL.write_text(updated)
    print(f"{GCC_BZL.name}: re-pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
