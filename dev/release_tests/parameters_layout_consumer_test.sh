#!/usr/bin/env bash
#===============================================================================
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
#===============================================================================

# Linux packaged-consumer smoke for a folded (BUILD_PARAMETERS_LIB=no) release.
set -euo pipefail

DALROOT="$(cd "${1:?usage: parameters_layout_consumer_test.sh DALROOT [source-root]}" && pwd)"
SOURCE_ROOT="$(cd "${2:-$(dirname "$0")/../..}" && pwd)"
BAZEL_CMD="${BAZEL:-bazel}"
work="$(mktemp -d)"

# The consumer workspace below is a throwaway directory, but the Bazel output
# base it gets is not: Bazel derives the output base from a hash of the
# workspace path, so it lands in its own directory under
# `~/.cache/bazel/_bazel_$USER/` and survives `rm -rf "${work}"`. It also
# survives the `bazel clean --expunge` steps around this test in
# .ci/pipeline/ci.yml, which expunge the *oneDAL* workspace's output base and
# know nothing about this one. Left behind, it costs about a gigabyte and a
# stranded Bazel server for the rest of the job -- on an agent that already
# reports over 95% of / in use, that is enough to make a later link fail with
# nothing but `collect2: error: ld returned 1 exit status`.
#
# Expunge it while the workspace still exists, before removing the workspace.
# `clean --expunge` also stops the server, so nothing keeps holding the files.
cleanup() {
    if [ -f "${work}/MODULE.bazel" ]; then
        (cd "${work}" && "${BAZEL_CMD}" clean --expunge) >/dev/null 2>&1 || true
    fi
    rm -rf "${work}"
}
trap cleanup EXIT

# `! producer | grep -q` is also satisfied when the *producer* fails, so an
# absent lib/ directory or an unmatched .pc glob would pass these three checks
# without looking at anything. Materialise the inputs first, then assert.
find "${DALROOT}/lib" \( -type f -o -type l \) -print >"${work}/lib-entries"
if grep 'onedal_parameters' "${work}/lib-entries"; then
    echo "folded release still ships a parameters library (above)" >&2
    exit 1
fi

config="${DALROOT}/lib/cmake/oneDAL/oneDALConfig.cmake"
if ! grep -Fq 'set(ONEDAL_USE_PARAMETERS_LIBRARY "no")' "${config}"; then
    echo "${config} does not advertise the folded layout; it says:" >&2
    grep -F 'ONEDAL_USE_PARAMETERS_LIBRARY' "${config}" >&2 || echo "  (no such line)" >&2
    exit 1
fi

# Report, rather than assert, the oneMKL references left undefined in the folded
# host library. The generated CMake config sets MKL_DEPENDENCY for this layout on
# non-Windows, and this is the evidence for whether that is necessary: these are
# the symbols a consumer of the folded package has to resolve itself and a
# consumer of the separate package resolves through libonedal_parameters. Not an
# assertion, because the correct count is whatever the layout implies, not a
# number this test should pin.
if command -v nm >/dev/null 2>&1; then
    host_lib="${DALROOT}/lib/intel64/libonedal.so"
    if [[ -e "${host_lib}" ]]; then
        echo "Undefined oneMKL references in $(basename "${host_lib}"):"
        # `grep -c` exits 1 on a count of zero, which is a valid answer here.
        nm -D --undefined-only "${host_lib}" 2>/dev/null \
            | grep -ciE 'mkl|_dgemm|_sgemm' || true
    fi
fi

pc_files=("${DALROOT}/lib/pkgconfig/"*.pc)
if [[ ! -e "${pc_files[0]}" ]]; then
    echo "no pkg-config files staged under ${DALROOT}/lib/pkgconfig" >&2
    exit 1
fi
if grep -H 'onedal_parameters' "${pc_files[@]}"; then
    echo "folded release still advertises a parameters library in pkg-config (above)" >&2
    exit 1
fi

cat >"${work}/smoke.cpp" <<'CPP'
#include "oneapi/dal/algo/covariance.hpp"
#include "oneapi/dal/table/homogen.hpp"

int main() {
    namespace dal = oneapi::dal;
    float data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const auto input = dal::homogen_table::wrap(data, 2, 2);
    const auto descriptor = dal::covariance::descriptor<float>{}.set_result_options(
        dal::covariance::result_options::cov_matrix);
    const auto result = dal::compute(descriptor, input);
    const auto& covariance = result.get_cov_matrix();
    return covariance.get_row_count() == 2 && covariance.get_column_count() == 2 ? 0 : 1;
}
CPP

cat >"${work}/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.16)
project(onedal_folded_consumer LANGUAGES CXX)
find_package(oneDAL REQUIRED CONFIG)
add_executable(smoke smoke.cpp)
if(UNIX AND NOT APPLE)
    target_link_libraries(smoke PRIVATE -Wl,--start-group ${oneDAL_IMPORTED_TARGETS} -Wl,--end-group)
else()
    target_link_libraries(smoke PRIVATE ${oneDAL_IMPORTED_TARGETS})
endif()
CMAKE
cmake_dependency_args=()
if [[ -n "${TBBROOT:-}" && -f "${TBBROOT}/lib/cmake/tbb/TBBConfig.cmake" ]]; then
    cmake_dependency_args+=("-DTBB_DIR=${TBBROOT}/lib/cmake/tbb")
fi
for mode in static dynamic; do
    cmake -S "${work}" -B "${work}/cmake-${mode}" \
        -DoneDAL_DIR="${DALROOT}/lib/cmake/oneDAL" \
        -DONEDAL_LINK="${mode}" -DONEDAL_USE_DPCPP=no -DONEDAL_INTERFACE=yes \
        "${cmake_dependency_args[@]}"
    cmake --build "${work}/cmake-${mode}" --parallel 2
    LD_LIBRARY_PATH="${DALROOT}/lib/intel64:${LD_LIBRARY_PATH:-}" "${work}/cmake-${mode}/smoke"
done

export PKG_CONFIG_PATH="${DALROOT}/lib/pkgconfig"
if [[ -n "${MKLROOT:-}" ]]; then
    export LIBRARY_PATH="${MKLROOT}/lib:${LIBRARY_PATH:-}"
fi
if [[ -n "${TBBROOT:-}" ]]; then
    export LIBRARY_PATH="${TBBROOT}/lib:${LIBRARY_PATH:-}"
fi
for pc in dal-dynamic-threading-host dal-static-threading-host; do
    # Resolved in its own statement: inside `c++ ... $(pkg-config ...)` a failing
    # substitution is not the command's own status, so `set -e` would let the
    # compile proceed with no flags and report a wall of undefined references
    # instead of the missing .pc.
    pc_flags="$(pkg-config --cflags --libs "${pc}")"
    # shellcheck disable=SC2086 # deliberate word splitting of the flag list
    c++ "${work}/smoke.cpp" -o "${work}/pkg-${pc}" ${pc_flags}
    LD_LIBRARY_PATH="${DALROOT}/lib/intel64:${LD_LIBRARY_PATH:-}" "${work}/pkg-${pc}"
done

cat >"${work}/MODULE.bazel" <<EOF
module(name = "onedal_folded_consumer")
bazel_dep(name = "onedal", version = "0.0.0")
bazel_dep(name = "rules_cc", version = "0.2.22")
local_path_override(module_name = "onedal", path = "${SOURCE_ROOT}")
tbb_repo = use_repo_rule("@onedal//dev/bazel/deps:tbb.bzl", "tbb_repo")
tbb_repo(name = "tbb", root_env_var = "TBBROOT")
mkl_repo = use_repo_rule("@onedal//dev/bazel/deps:mkl.bzl", "mkl_repo")
mkl_repo(name = "mkl", root_env_var = "MKLROOT")
onedal_repo = use_repo_rule("@onedal//dev/bazel/deps:onedal.bzl", "onedal_repo")
onedal_repo(name = "onedal_release", root_env_var = "DALROOT")
EOF
cat >"${work}/BUILD" <<'BUILD'
load("@rules_cc//cc:defs.bzl", "cc_binary")
cc_binary(
    name = "static",
    srcs = ["smoke.cpp"],
    deps = [
        "@onedal_release//:onedal_static",
        "@onedal_release//:core_static",
        "@onedal_release//:thread_static",
    ],
)
cc_binary(
    name = "dynamic",
    srcs = ["smoke.cpp"],
    deps = [
        "@onedal_release//:onedal_dynamic",
        "@onedal_release//:core_dynamic",
        "@onedal_release//:thread_dynamic",
    ],
)
BUILD
(
    cd "${work}"
    DALROOT="${DALROOT}" "${BAZEL_CMD}" build //:static //:dynamic --noshow_progress --show_result=0
    DALROOT="${DALROOT}" "${BAZEL_CMD}" run //:static --noshow_progress --show_result=0
    DALROOT="${DALROOT}" "${BAZEL_CMD}" run //:dynamic --noshow_progress --show_result=0
)

echo "Folded packaged CMake, pkg-config, and Bazel consumers passed"
