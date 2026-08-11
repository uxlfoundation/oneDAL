#===============================================================================
# Copyright 2026 Intel Corporation
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
Builds and runs a CMake example against the Bazel-produced Windows release.

.DESCRIPTION
This is a consumer smoke test for the packaged release, not another Bazel
build. It points the examples' CMake project at the release through
`CMAKE_PREFIX_PATH`, builds one dynamically linked target, and runs it. This
checks that headers, CMake package metadata, import libraries, and runtime DLLs
from `//:release` work together for a downstream CMake consumer.

The script removes the examples' `Build` directory to prevent a cached CMake
configuration from hiding packaging problems. It accepts either the oneAPI
installation or CI-downloaded TBB layout and fails on configuration, build,
missing-executable, or runtime errors.

.PARAMETER ReleaseDir
Root of the unpacked or freshly built oneDAL release package.

.PARAMETER ExamplesDir
Examples directory relative to the release root.

.PARAMETER ExampleSource
Reserved for selecting an example source; the current CMake flow selects the
target through `ExampleTarget`.

.PARAMETER ExampleTarget
CMake example target to configure, build, locate, and run.
#>

param(
    [string]$ReleaseDir = ".\bazel-bin\release\daal\latest",
    [string]$ExamplesDir = "examples\oneapi\cpp",
    [string]$ExampleSource = "source\basic_statistics\basic_statistics_dense_batch.cpp",
    [string]$ExampleTarget = "basic_statistics_dense_batch"
)

$ErrorActionPreference = "Stop"

$releasePath = (Resolve-Path $ReleaseDir).Path
$examplesPath = Join-Path $releasePath $ExamplesDir
$buildPath = Join-Path $examplesPath "Build"

if (Test-Path $buildPath) {
    # A clean configure proves that the package is self-consistent.
    Remove-Item -Recurse -Force $buildPath
}

$pathEntries = @(
    (Join-Path $releasePath "redist\intel64"),
    (Join-Path $releasePath "lib")
)

$tbbDirCandidates = @(
    # Developer installation first, then the dependency layout used in CI.
    ".\oneapi\tbb\latest\lib\cmake\tbb",
    ".\__deps\tbb\win\tbb\lib\cmake\tbb"
)
$tbbRedistCandidates = @(
    ".\oneapi\tbb\latest\redist\intel64\vc_mt",
    ".\__deps\tbb\win\tbb\redist\intel64\vc_mt"
)

$cmakeArgs = @(
    "-B", $buildPath,
    "-S", $examplesPath,
    "-DONEDAL_LINK=dynamic",
    "-DCMAKE_PREFIX_PATH=$releasePath",
    "-DEXAMPLES_LIST=$ExampleTarget"
)

foreach ($candidate in $tbbDirCandidates) {
    if (Test-Path $candidate) {
        $cmakeArgs += "-DTBB_DIR=$((Resolve-Path $candidate).Path)"
        break
    }
}

foreach ($candidate in $tbbRedistCandidates) {
    if (Test-Path $candidate) {
        $pathEntries += (Resolve-Path $candidate).Path
        break
    }
}

$env:PATH = (($pathEntries | Where-Object { Test-Path $_ }) -join ";") + ";" + $env:PATH

# Dynamic oneDAL examples need packaged oneDAL DLLs and a TBB runtime on PATH.

Write-Host "Configuring CMake example from Bazel release: $examplesPath"
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed with exit code $LASTEXITCODE"
}

Write-Host "Building CMake example target: $ExampleTarget"
cmake --build $buildPath --config Release --target $ExampleTarget
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed with exit code $LASTEXITCODE"
}

$exampleExe = Get-ChildItem -Path @($buildPath, $examplesPath) -Recurse -Filter "$ExampleTarget.exe" -File |
    Select-Object -First 1
if ($null -eq $exampleExe) {
    Write-Host "Available executables under examples directory:"
    Get-ChildItem -Path $examplesPath -Recurse -Filter "*.exe" -File | ForEach-Object { Write-Host "  $($_.FullName)" }
    throw "Cannot find built example executable: $ExampleTarget.exe"
}

Write-Host "Running CMake example: $($exampleExe.FullName)"
Push-Location $examplesPath
try {
    & $exampleExe.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Example failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
