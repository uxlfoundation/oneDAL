#!/usr/bin/env bash
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

# Usage: run.sh NAME L B 'MUTSET'   (needs java and tla2tools.jar; TLA2TOOLS env var)
set -euo pipefail
jar=$(realpath "${TLA2TOOLS:?set TLA2TOOLS to tla2tools.jar}")
cd "$(dirname "$0")"
name=$1; l=$2; b=$3; mut=$4
mkdir -p out
cat > "out/$name.cfg" <<CFG
CONSTANTS L = $l
          B = $b
          Mut = $mut
SPECIFICATION Spec
INVARIANTS NoUseAfterFree NoReadBeforeWrite NoStaleLocalRead ResultCoversAllBlocks ResultReadAfterK2
PROPERTY Termination
CFG
java -XX:+UseParallelGC -cp "$jar" tlc2.TLC -workers 8 -config "out/$name.cfg" \
  -metadir "out/$name.states" MergeBlocks.tla > "out/$name.log" 2>&1 && rc=0 || rc=$?
# A mutation run is expected to stop with a violation (non-zero exit); print the
# verdict either way and return TLC's exit status.
if ! grep -E "Model checking completed|violated|Error:|distinct states found|Deadlock" "out/$name.log" | head -4; then
  echo "no TLC verdict line in out/$name.log"
fi
echo "tlc exit status: $rc"
exit "$rc"
