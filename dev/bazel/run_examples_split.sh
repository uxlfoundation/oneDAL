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
# Build and run oneDAL Bazel examples in two explicit phases.
#
# Usage:
#   dev/bazel/run_examples_split.sh build [host|dpc|all] [bazel targets/packages...]
#   dev/bazel/run_examples_split.sh run   [host|dpc|all] [bazel targets/packages...]
#   dev/bazel/run_examples_split.sh all   [host|dpc|all] [bazel targets/packages...]
#   dev/bazel/run_examples_split.sh list  [host|dpc|all] [bazel targets/packages...]
#
# Examples:
#   dev/bazel/run_examples_split.sh build host //examples/...
#   dev/bazel/run_examples_split.sh run host //examples/...
#   dev/bazel/run_examples_split.sh all dpc //examples/oneapi/dpc/...
#
# Extra Bazel flags can be passed through BAZEL_FLAGS / BUILD_FLAGS / TEST_FLAGS, e.g.:
#   BAZEL_FLAGS="--config=dpc" TEST_FLAGS="--local_test_jobs=8 --test_output=errors" \
#     dev/bazel/run_examples_split.sh all dpc //examples/oneapi/dpc/...

set -euo pipefail

MODE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi

KIND="${1:-all}"
case "${KIND}" in
    host|dpc|all)
        if [[ $# -gt 0 ]]; then shift; fi
        ;;
    *)
        KIND="all"
        ;;
esac

if [[ $# -gt 0 ]]; then
    QUERY_TARGETS=("$@")
else
    QUERY_TARGETS=("//examples/...")
fi

case "${MODE}" in
    build|run|all|list) ;;
    *)
        echo "Unknown mode '${MODE}'. Expected: build, run, all, list" >&2
        exit 2
        ;;
esac

kind_filter() {
    case "${KIND}" in
        host) grep -E '_host$' ;;
        dpc)  grep -E '_dpc$' ;;
        all)  cat ;;
    esac
}

collect_targets() {
    local query_parts=()
    local target
    for target in "${QUERY_TARGETS[@]}"; do
        query_parts+=("tests(${target})")
    done
    local query_expr="${query_parts[0]}"
    local i
    for ((i = 1; i < ${#query_parts[@]}; i++)); do
        query_expr="${query_expr} + ${query_parts[$i]}"
    done

    "${BAZEL_BIN:-bazel}" query "${query_expr}" \
        | kind_filter \
        | sort -u
}

detect_jobs() {
    if [[ -n "${BAZEL_JOBS:-}" ]]; then
        echo "${BAZEL_JOBS}"
    elif [[ -n "${NUMBER_OF_PROCESSORS:-}" ]]; then
        echo "${NUMBER_OF_PROCESSORS}"
    elif command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v getconf >/dev/null 2>&1; then
        getconf _NPROCESSORS_ONLN
    else
        echo 1
    fi
}

BAZEL_JOBS="$(detect_jobs)"
BAZEL_BIN="${BAZEL:-bazel}"

mapfile -t TARGETS < <(collect_targets)

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "No example test targets found for kind='${KIND}' in: ${QUERY_TARGETS[*]}" >&2
    exit 1
fi

echo "Selected ${#TARGETS[@]} example targets (${KIND}) from: ${QUERY_TARGETS[*]}" >&2
echo "Using ${BAZEL_JOBS} parallel Bazel jobs/test jobs" >&2
echo "Using Bazel binary: ${BAZEL_BIN}" >&2

if [[ "${MODE}" == "list" ]]; then
    printf '%s\n' "${TARGETS[@]}"
    exit 0
fi

build_examples() {
    echo "==> BUILD phase: compiling/linking all selected examples" >&2
    # shellcheck disable=SC2086
    "${BAZEL_BIN}" build --jobs="${BAZEL_JOBS}" ${BAZEL_FLAGS:-} ${BUILD_FLAGS:-} "${TARGETS[@]}"
}

TEST_ENV_PATH_FLAG=()

configure_windows_runtime_path() {
    case "$(uname -s 2>/dev/null || echo unknown)" in
        MINGW*|MSYS*|CYGWIN*)
            local output_base output_base_win path_win
            output_base="$("${BAZEL_BIN}" info output_base 2>/dev/null || true)"
            if [[ -n "${output_base}" ]]; then
                export PATH="${output_base}/external/+mkl_repo+mkl/bin:${output_base}/external/+tbb_repo+tbb/bin:${output_base}/external/+mkl_repo+mkl/archive/bin:${output_base}/external/+tbb_repo+tbb/archive/bin:${PATH}"
                output_base_win="$(cygpath -w "${output_base}" 2>/dev/null || echo "${output_base}")"
                path_win="$(cygpath -wp "${PATH}" 2>/dev/null || echo "${PATH}")"
                export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-TBB}"
                TEST_ENV_PATH_FLAG=(
                    "--test_env=PATH=${output_base_win}\\external\\+mkl_repo+mkl\\bin;${output_base_win}\\external\\+tbb_repo+tbb\\bin;${output_base_win}\\external\\+mkl_repo+mkl\\archive\\bin;${output_base_win}\\external\\+tbb_repo+tbb\\archive\\bin;${path_win}"
                    "--test_env=MKL_THREADING_LAYER=${MKL_THREADING_LAYER}"
                    "--test_env=DALROOT=$(pwd)"
                )
            fi
            ;;
    esac
}

run_examples() {
    echo "==> RUN phase: running already-built examples as Bazel tests" >&2
    configure_windows_runtime_path
    # --build_tests_only keeps Bazel from building non-test dependencies outside the selected tests.
    # After the build phase, this should mostly schedule test actions, while still staying correct
    # if a target changed between phases.
    # shellcheck disable=SC2086
    "${BAZEL_BIN}" test --build_tests_only --test_output=errors "${TEST_ENV_PATH_FLAG[@]}" --jobs="${BAZEL_JOBS}" --local_test_jobs="${BAZEL_JOBS}" ${BAZEL_FLAGS:-} ${TEST_FLAGS:-} "${TARGETS[@]}"
}

case "${MODE}" in
    build)
        build_examples
        ;;
    run)
        run_examples
        ;;
    all)
        build_examples
        run_examples
        ;;
esac
