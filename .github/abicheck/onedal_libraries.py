#!/usr/bin/env python3
# ******************************************************************************
# Copyright contributors to the oneDAL project
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
# ******************************************************************************

"""oneDAL's per-library ABI scoping table — the one source of truth for it.

Imported by both halves of ``bundle_gate.py``: the baseline ``capture`` and the
PR-side ``gate`` read this same table, so the two sides cannot disagree about
which headers define a library's public surface. A baseline captured from a
different header set than the gate uses reports every header-only difference as
a spurious add/remove, which is why this lives in one place rather than being
passed in per workflow step.

Split out from ``bundle_gate.py`` only to keep that file under this
repository's 500-line limit; it has no other consumer.
"""

from __future__ import annotations

# oneDAL exports three independent public surfaces plus two parameters
# libraries; libonedal_thread.so has no installed header at all (its _daal_*
# threading entry points are declared only in cpp/daal/src/threading/, which is
# not shipped), so it is carried ELF-only. -DONEDAL_DATA_PARALLEL is what makes
# the sycl::queue overloads visible, and it is exactly the part of the API only
# the _dpc libraries export.
_DAAL_HEADERS = [
    "cpp/daal/include/daal.h",
    "cpp/daal/include/algorithms/tsne/tsne_gradient_descent.h",
    "cpp/daal/include/data_management/data/internal/finiteness_checker.h",
    "cpp/daal/include/data_management/data/internal/roc_auc_score.h",
    "cpp/daal/include/data_management/data/internal/train_test_split.h",
    "cpp/daal/include/services/internal/hash_table.h",
    "cpp/daal/include/data_management/data/internal/base_arrow_numeric_table.h",
]

_ONEAPI_HEADERS = [
    "cpp/oneapi/dal.hpp",
    "cpp/oneapi/dal/algo/correlation_distance.hpp",
    "cpp/oneapi/dal/algo/cosine_distance.hpp",
    "cpp/oneapi/dal/algo/dbscan.hpp",
    "cpp/oneapi/dal/algo/linear_regression.hpp",
    "cpp/oneapi/dal/algo/logistic_regression.hpp",
    "cpp/oneapi/dal/table/csr_accessor.hpp",
]

# The parameters libraries' only *consumable* public header. oneDAL installs 14
# parameters headers, but the 12 algo ones
# (oneapi/dal/algo/*/parameters/{cpu,gpu}/*.hpp) all include
# "oneapi/dal/backend/dispatcher.hpp", and oneapi/dal/backend/ is not installed
# at all -- verified 2026-08-24 by compiling each installed header against the
# release include tree alone: 12 of 14 fail with `'oneapi/dal/backend/
# dispatcher.hpp' file not found`, and from the *source* tree they instead fail
# on backend/dispatcher_cpu.hpp's build-generated
# $(WORKDIR)/oneapi/dal/_dal_cpu_dispatcher_gen.hpp. So no consumer can include
# them as shipped, and header-scoping to them would scope against a surface
# nobody can reach. Left out deliberately, not overlooked; the packaging gap is
# an oneDAL bug independent of abicheck. system_parameters.hpp needs only
# detail/ headers, all of which are installed, so it stays in.
_PARAM_HEADERS = ["cpp/oneapi/dal/detail/parameters/system_parameters.hpp"]

_STD = "-std=c++17"
_SYCL = "-fsycl"
_DP = "-DONEDAL_DATA_PARALLEL"

#: canonical abicheck library key -> (headers, include roots, compiler options).
#: The keys are what ``cli_helpers_compare._build_match_map`` derives from the
#: real filenames (libonedal.so.4.0 -> libonedal.so), so they are what
#: ``BundleFacts.per_library_snapshots`` must be keyed by.
LIBRARIES: dict[str, dict[str, list[str]]] = {
    "libonedal_core.so": {
        "headers": _DAAL_HEADERS,
        "includes": ["cpp/daal/include"],
        "options": [_STD],
    },
    "libonedal.so": {
        "headers": _ONEAPI_HEADERS,
        "includes": ["cpp"],
        "options": [_SYCL, _STD],
    },
    "libonedal_dpc.so": {
        "headers": _ONEAPI_HEADERS,
        "includes": ["cpp"],
        "options": [_SYCL, _DP, _STD],
    },
    "libonedal_parameters.so": {
        "headers": _PARAM_HEADERS,
        "includes": ["cpp"],
        "options": [_SYCL, _STD],
    },
    "libonedal_parameters_dpc.so": {
        "headers": _PARAM_HEADERS,
        "includes": ["cpp"],
        "options": [_SYCL, _DP, _STD],
    },
    # No installed public header: ELF-only, by design rather than as a gap.
    "libonedal_thread.so": {"headers": [], "includes": [], "options": []},
}

#: DT_NEEDED sonames that are external to the product. abicheck's own
#: DEFAULT_SYSTEM_PROVIDERS omits tbbmalloc, the MKL libraries and the Intel
#: runtime, and one unmatched edge disables the system-edge exemption for a
#: whole library, so this list is load-bearing rather than belt-and-braces.
SYSTEM_PROVIDERS = [
    "libm", "libgcc_s", "libc", "ld-linux-x86-64", "libstdc++", "libgomp",
    "libsycl", "libmkl_sycl_blas", "libmkl_sycl_lapack", "libmkl_sycl_sparse",
    "libmkl_sycl_rng", "libmkl_core", "libmkl_intel_lp64", "libmkl_gnu_thread",
    "libimf", "libsvml", "libirng", "libintlc", "libtbb", "libtbbmalloc",
    "libpthread", "libdl", "librt",
]
