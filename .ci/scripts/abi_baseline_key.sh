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
# Usage: abi_baseline_key.sh <base-commit> [window]
#
# Taking the newest entry in main's cache scope is wrong for three reasons.
#
# Scope. `gh cache list` shows every scope in the repository, but
# `cache/restore` can only read main's. An entry from another pull request's
# scope would miss, and the comparison would then run against an empty
# directory.
#
# Provenance. Candidates come from commits, not from the cache listing, so the
# newest key in the listing does not decide anything. Only trusted triggers
# (push, schedule, workflow_dispatch) can write to main's cache scope; runs from
# workflow_run, pull_request_target and issue_comment get read-only access. Even
# so, a key for a real main commit whose entry has expired could be re-created
# by anything that can write there, and the walk below would pick it if nothing
# newer is cached. The restored tree is only ever read by abidiff.
#
# Ancestry. abidiff compares two trees; it does not report what a pull request
# changed. So start from the base commit the build contains: when that commit is
# cached, the report holds the pull request's own changes and nothing else. When
# it is not (its main CI run has not finished saving the cache yet), the newest
# cached ancestor is used and the main commits in between show up in the report
# as if the pull request had made them. A change that undoes one of them -- such
# as removing a symbol main just added -- then cancels out and is not reported.
# Every inexact comparison says so in a warning.

set -euo pipefail

BASE_SHA=${1:-}
WINDOW=${2:-20}
KEY_PREFIX=__release_lnx

# An empty base is always a wiring bug in the caller (for example reading the
# parents of a shallow clone's HEAD, which git reports as having none), never a
# cache gap, so do not let it fall through to the inexact baseline.
if [ -z "${BASE_SHA}" ]; then
    echo "::error::No base commit was passed. The caller must pass the first parent of the merge commit under test."
    exit 1
fi

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

available=$(retry gh cache list --repo "${GITHUB_REPOSITORY}" --ref refs/heads/main --key "${KEY_PREFIX}" --limit 100 --json key --jq '.[].key') || {
    echo "::error::Could not list the caches on refs/heads/main. Rerun this job."
    exit 1
}

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

base_commits=$(ancestors_of "${BASE_SHA}") || {
    echo "::error::Could not list the commits reachable from ${BASE_SHA}. Rerun this job."
    exit 1
}
exact=yes
result=$(newest_cached_in "${base_commits}") || result=""
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
elif [ "${behind}" -gt 0 ]; then
    exact=no
    echo "::warning::The base commit's main build is not cached yet, so the baseline is ${behind} commit(s) behind it. Those commits' ABI changes show up in this check, and a change in this pull request that undoes one of them is not reported. Rerun this job once the 'CI' run on main for the base commit has finished."
fi

echo "${cache_key} (exact: ${exact}, ${behind} commits back)"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "key=${cache_key}" >> "${GITHUB_OUTPUT}"
fi
