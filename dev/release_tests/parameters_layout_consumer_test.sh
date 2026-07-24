#!/usr/bin/env bash
# Linux packaged-consumer smoke for a folded (BUILD_PARAMETERS_LIB=no) release.
set -euo pipefail

DALROOT="$(cd "${1:?usage: parameters_layout_consumer_test.sh DALROOT [source-root]}" && pwd)"
SOURCE_ROOT="$(cd "${2:-$(dirname "$0")/../..}" && pwd)"
BAZEL_CMD="${BAZEL:-bazel}"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

! find "${DALROOT}/lib" -type f -o -type l | grep -q 'onedal_parameters'
grep -Fq 'set(ONEDAL_USE_PARAMETERS_LIBRARY "no")' "${DALROOT}/lib/cmake/oneDAL/oneDALConfig.cmake"
! grep -q 'onedal_parameters' "${DALROOT}/lib/pkgconfig/"*.pc

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
    c++ "${work}/smoke.cpp" -o "${work}/pkg-${pc}" $(pkg-config --cflags --libs "${pc}")
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
