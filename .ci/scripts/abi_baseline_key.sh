#!/bin/bash
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

# Writes to GITHUB_OUTPUT the cache key of the main build the ABI check compares
# against: the newest cached ancestor of the pull request's base commit.
#
# Usage: abi_baseline_key.sh <base-commit> [window]
#
# abidiff compares two trees, so the baseline must be an ancestor of the build
# under test; main's tip may already contain changes the build does not.

set -euo pipefail

BASE_SHA=${1:-}
WINDOW=${2:-20}
KEY_PREFIX=__release_lnx

# An empty base is a caller bug (a shallow clone's HEAD reports no parents).
if [ -z "${BASE_SHA}" ]; then
    echo "::error::No base commit was passed. The caller must pass the first parent of the merge commit under test."
    exit 1
fi

# Runs "$@" up to three times and echoes its output.
retry() {
    local delay out
    for delay in 0 5 10; do
        sleep "${delay}"
        if out=$("$@"); then
            echo "${out}"
            return 0
        fi
    done
    return 1
}

available=$(retry gh cache list --repo "${GITHUB_REPOSITORY}" --ref refs/heads/main --key "${KEY_PREFIX}" --limit 100 --json key --jq '.[].key') || {
    echo "::error::Could not list the caches on refs/heads/main. Rerun this job."
    exit 1
}

# A failure would look like "no cached ancestor", so the caller stops the job.
ancestors_of() {
    retry gh api "repos/${GITHUB_REPOSITORY}/commits?sha=$1&per_page=${WINDOW}" --jq '.[].sha'
}

# Echoes "<key> <commits skipped>" for the newest commit in $1 that has an
# entry, or returns 1 if none of them does.
newest_cached_in() {
    local behind=0 sha
    for sha in $1; do
        if grep -qxF "${KEY_PREFIX}-${sha}" <<< "${available}"; then
            echo "${KEY_PREFIX}-${sha} ${behind}"
            return 0
        fi
        behind=$((behind + 1))
    done
    return 1
}

base_commits=$(ancestors_of "${BASE_SHA}") || {
    echo "::error::Could not list the commits reachable from ${BASE_SHA}. Rerun this job."
    exit 1
}
result=$(newest_cached_in "${base_commits}") || {
    echo "::error::No cached main build for this pull request's base or its last ${WINDOW} ancestors to compare against. Rebase onto main and rerun."
    exit 1
}
read -r cache_key behind <<< "${result}"

exact=yes
if [ "${behind}" -gt 0 ]; then
    exact=no
    echo "::warning::The base commit's main build is not cached yet, so the baseline is ${behind} commit(s) behind it. Those commits' ABI changes show up in this check, and a change in this pull request that undoes one of them is not reported. Rerun this job once the 'CI' run on main for the base commit has finished."
fi

echo "${cache_key} (exact: ${exact}, ${behind} commits back)"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "key=${cache_key}" >> "${GITHUB_OUTPUT}"
fi
