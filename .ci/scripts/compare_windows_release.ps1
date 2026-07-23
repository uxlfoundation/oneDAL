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

<#
.SYNOPSIS
Compares the file and binary surfaces of Make and Bazel Windows releases.

.DESCRIPTION
The nightly Windows job builds the existing Make `vc` release and a Bazel
release under the same oneAPI and Visual Studio environment, then calls this
adapter to run `dev/release_tests/compare_release_trees.py` with Windows
semantics. Bazel normally auto-selects its ICX-based Windows toolchain, so this
checks package and public binary-surface compatibility, not compiler
equivalence or compiler-specific intermediate objects.

All paths may be relative to the current working directory. The Python tool
prints a bounded summary and exits nonzero when the selected check level finds
a mismatch, which causes the calling CI step to fail.

.PARAMETER MakeReleaseDir
Path to the reference release produced by the Make-based Windows build.

.PARAMETER BazelReleaseDir
Path to the release produced by the Bazel `//:release` target.

.PARAMETER CheckLevel
Validation depth passed to `compare_release_trees.py`; level 4 includes the
full manifest and binary-surface checks used by nightly CI.

.PARAMETER MaxDiffLines
Maximum number of difference lines printed in the summary.
#>

param(
    [string] $MakeReleaseDir = "__release_win_vc\daal\latest",
    [string] $BazelReleaseDir = "bazel-bin\release\daal\latest",
    [int] $CheckLevel = 4,
    [int] $MaxDiffLines = 200
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")
$compareScript = Join-Path $repoRoot "dev\release_tests\compare_release_trees.py"

python $compareScript `
    --platform windows `
    --make $MakeReleaseDir `
    --bazel $BazelReleaseDir `
    --check-level $CheckLevel `
    --summary-limit $MaxDiffLines

# Windows PowerShell 5.1 does not turn native-command failures into terminating
# errors, even when ErrorActionPreference is Stop.
exit $LASTEXITCODE
