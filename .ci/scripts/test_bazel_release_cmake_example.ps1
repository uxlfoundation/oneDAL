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
    Remove-Item -Recurse -Force $buildPath
}

$pathEntries = @(
    (Join-Path $releasePath "redist\intel64"),
    (Join-Path $releasePath "lib")
)

$tbbDirCandidates = @(
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
    "-DEXAMPLES_LIST=$ExampleSource"
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

Write-Host "Configuring CMake example from Bazel release: $examplesPath"
cmake @cmakeArgs

Write-Host "Building CMake example target: $ExampleTarget"
cmake --build $buildPath --config Release

$exe = Get-ChildItem -Path $examplesPath -Filter "$ExampleTarget.exe" -Recurse | Select-Object -First 1
if (-not $exe) {
    throw "Cannot find built example executable: $ExampleTarget.exe"
}

Write-Host "Running CMake example: $($exe.FullName)"
Push-Location $examplesPath
try {
    & $exe.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Example failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
