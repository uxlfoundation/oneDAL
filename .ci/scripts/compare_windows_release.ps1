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
