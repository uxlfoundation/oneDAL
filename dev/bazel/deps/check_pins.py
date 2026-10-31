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
"""Fails when the Make and Bazel builds pin different oneMKL or oneTBB releases.

Renovate groups each library's apt package and wheels into one PR, but a group
only bundles the updates that are published at the time: when PyPI has a
release that apt.repos.intel.com does not have yet (or the other way round),
the PR moves one build only. This check makes such a PR fail until both sides
are on one release. It also keeps the wheels of one library on one version.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "dev/bazel/deps/requirements_lock.txt"
APT = ROOT / ".ci/env/apt.sh"
TBB_SH = ROOT / ".ci/env/tbb.sh"

# Library -> (lock packages that must share one version, apt package).
LIBRARIES = {
    "oneMKL": (
        [
            "mkl",
            "mkl-static",
            "mkl-include",
            "mkl-devel",
            "onemkl-sycl-include",
            "onemkl-sycl-blas",
            "onemkl-sycl-lapack",
            "onemkl-sycl-sparse",
            "onemkl-sycl-rng",
        ],
        "intel-oneapi-mkl-devel",
    ),
    "oneTBB": (["tbb", "tbb-devel"], "intel-oneapi-tbb-devel"),
    "Intel MPI": (["impi-rt", "impi-devel"], None),
}


def lock_versions():
    pins = re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", LOCK.read_text(), re.MULTILINE)
    return {name.lower().replace("_", "-"): version for name, version in pins}


def main():
    pins = lock_versions()
    apt = APT.read_text()
    errors = []
    for library, (packages, apt_package) in LIBRARIES.items():
        versions = {p: pins[p] for p in packages if p in pins}
        if len(set(versions.values())) > 1:
            errors.append(f"{library}: the wheels in {LOCK.name} disagree: {versions}")
            continue
        if not versions:
            errors.append(f"{library}: none of {packages} is pinned in {LOCK.name}")
            continue
        if not apt_package:
            continue
        version = next(iter(versions.values()))
        match = re.search(rf"\b{apt_package}=(\d+\.\d+\.\d+)-\d+", apt)
        if not match:
            errors.append(f"{library}: no pinned {apt_package}= in {APT.name}")
        elif match[1] != version:
            errors.append(
                f"{library}: Make installs {match[1]} ({APT.name}), "
                f"Bazel downloads {version} ({LOCK.name})"
            )
    tbb_tag = re.search(r"^TBB_DEFAULT_VERSION=v?(\S+)", TBB_SH.read_text(), re.MULTILINE)
    if not tbb_tag:
        errors.append(f"oneTBB: no TBB_DEFAULT_VERSION= in {TBB_SH.name}")
    elif pins.get("tbb") and tbb_tag[1] != pins["tbb"]:
        errors.append(
            f"oneTBB: {TBB_SH.name} builds {tbb_tag[1]}, Bazel downloads {pins['tbb']} ({LOCK.name})"
        )
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if not errors:
        print("Make and Bazel pin the same oneMKL and oneTBB releases")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
