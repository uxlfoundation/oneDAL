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
    [int] $MaxDiffLines = 200
)

$ErrorActionPreference = "Stop"

function Resolve-ReleaseDir {
    param([string] $Path)

    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue
    if (-not $resolved) {
        throw "Release directory does not exist: $Path"
    }
    return $resolved.ProviderPath.TrimEnd('\', '/')
}

function Get-ReleaseManifest {
    param([string] $Root)

    $rootPrefix = $Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    Get-ChildItem -LiteralPath $Root -Recurse -File |
        ForEach-Object {
            $_.FullName.Substring($rootPrefix.Length).Replace('\', '/')
        } |
        Sort-Object
}

function Write-DiffBlock {
    param(
        [string] $Title,
        [string[]] $Items
    )

    Write-Host ""
    Write-Host $Title
    if ($Items.Count -eq 0) {
        Write-Host "  none"
        return
    }

    $Items | Select-Object -First $MaxDiffLines | ForEach-Object {
        Write-Host "  $_"
    }

    if ($Items.Count -gt $MaxDiffLines) {
        Write-Host "  ... $($Items.Count - $MaxDiffLines) more"
    }
}

$makeRoot = Resolve-ReleaseDir $MakeReleaseDir
$bazelRoot = Resolve-ReleaseDir $BazelReleaseDir

Write-Host "Comparing Windows release manifests"
Write-Host "  make : $makeRoot"
Write-Host "  bazel: $bazelRoot"

$makeFiles = @(Get-ReleaseManifest $makeRoot)
$bazelFiles = @(Get-ReleaseManifest $bazelRoot)

$makeSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
$bazelSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

$makeFiles | ForEach-Object { [void] $makeSet.Add($_) }
$bazelFiles | ForEach-Object { [void] $bazelSet.Add($_) }

$missingFromBazel = @($makeFiles | Where-Object { -not $bazelSet.Contains($_) })
$extraInBazel = @($bazelFiles | Where-Object { -not $makeSet.Contains($_) })

Write-Host "  make files : $($makeFiles.Count)"
Write-Host "  bazel files: $($bazelFiles.Count)"

if ($missingFromBazel.Count -eq 0 -and $extraInBazel.Count -eq 0) {
    Write-Host "PASS: Windows make and Bazel release manifests match"
    exit 0
}

Write-DiffBlock "Files present in make release but missing from Bazel release:" $missingFromBazel
Write-DiffBlock "Files present in Bazel release but missing from make release:" $extraInBazel

Write-Host "FAIL: Windows make and Bazel release manifests differ"
exit 1
