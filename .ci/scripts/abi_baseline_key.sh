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

# Chooses the cached main build that the ABI check compares against and writes
# its cache key to GITHUB_OUTPUT.
#
# Usage: abi_baseline_key.sh <base-commit> [window] [stale-after]
#
# Taking the newest entry in main's cache scope is wrong for three reasons.
#
# Scope. `gh cache list` shows every scope in the repository, but
# `cache/restore` can only read main's. An entry from another pull request's
# scope would miss, and the comparison would then run against an empty
# directory.
#
# Provenance. Any job whose GITHUB_REF is main can write into main's cache
# scope, the fork pull request jobs in Nightly-test included, and a key that did
# not exist before is always the newest one. So candidates come from commits
# rather than from the cache listing. Keys are immutable, so the only race left
# is against the real main build for a real commit's key.
#
# Ancestry. abidiff compares two trees; it does not report what a pull request
# changed. When the baseline is not an ancestor of the build under test, main's
# own changes show up in the report inverted, and a change the pull request
# shares with one of them does not show up at all -- that is how a real break
# passes. So start from the base commit the build contains. Fall back to main's
# tip only when that base has no cached build left, and say that the comparison
# is no longer exact.

set -euo pipefail

BASE_SHA=${1:-}
WINDOW=${2:-20}
STALE_AFTER=${3:-5}
KEY_PREFIX=__release_lnx

# Runs "$@" up to three times, backing off in between, and echoes its output, so
# one API hiccup does not get to decide which baseline the check uses.
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

# A failure to list the caches is safe to carry on from: no key can match, so
# the job stops at the hard error below rather than comparing against nothing.
available=$(retry gh cache list --ref refs/heads/main --key "${KEY_PREFIX}" --limit 100 --json key --jq '.[].key') || available=""

# A failure here is not safe to carry on from. It yields no candidates, which is
# indistinguishable from "no cached ancestor" and would quietly pick the inexact
# baseline, so both callers stop the job.
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

result=""
exact=yes
if [ -n "${BASE_SHA}" ]; then
    base_commits=$(ancestors_of "${BASE_SHA}") || {
        echo "::error::Could not list the commits reachable from ${BASE_SHA}. Rerun this job."
        exit 1
    }
    result=$(newest_cached_in "${base_commits}") || result=""
fi
if [ -z "${result}" ]; then
    exact=no
    main_commits=$(ancestors_of main) || {
        echo "::error::Could not list the commits on refs/heads/main. Rerun this job."
        exit 1
    }
    result=$(newest_cached_in "${main_commits}") || {
        echo "::error::No ${KEY_PREFIX} cache on refs/heads/main for any of its last ${WINDOW} commits. Rerun the 'CI' workflow on main via workflow dispatch to regenerate it."
        exit 1
    }
fi
read -r cache_key behind <<< "${result}"

if [ "${exact}" = no ]; then
    echo "::warning::No cached main build for this pull request's base or its last ${WINDOW} ancestors, so the baseline is main's tip. Changes merged into main since the base will show up in this check, and a change this pull request shares with one of them will not show up at all. Rebase onto main for an exact comparison."
elif [ "${behind}" -gt "${STALE_AFTER}" ]; then
    echo "::warning::Baseline is ${behind} commits behind this pull request's base. Changes merged into main in between will show up in this check."
fi

echo "${cache_key} (exact: ${exact}, ${behind} commits back)"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "key=${cache_key}" >> "${GITHUB_OUTPUT}"
fi
