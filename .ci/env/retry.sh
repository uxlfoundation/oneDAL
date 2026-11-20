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

# Sourced by the .ci/env installers: `retry <command...>` reruns a failed network command
# with exponential backoff; RETRY_ATTEMPTS (5) and RETRY_DELAY (5 s, doubled) tune it.

retry() {
    local attempts=${RETRY_ATTEMPTS:-5}
    local delay=${RETRY_DELAY:-5}
    local attempt=1
    local status
    while true; do
        "$@" && return 0
        status=$?
        if [ "${attempt}" -ge "${attempts}" ]; then
            echo "retry: giving up on '$*' after ${attempt} attempts (exit ${status})" >&2
            return "${status}"
        fi
        echo "retry: attempt ${attempt} of ${attempts} for '$*' failed (exit ${status}), retrying in ${delay}s" >&2
        sleep "${delay}"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done
}
