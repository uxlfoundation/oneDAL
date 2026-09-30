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
Installs the pinned Windows Bazelisk executable for CI.

.DESCRIPTION
This is the PowerShell counterpart of `.ci/env/bazelisk.sh`; keep the pinned
version in step with it. The Windows jobs cannot use that POSIX-shell helper, so
this script downloads the Windows asset, verifies it against the digest pinned
below, and makes `bazel.exe` available to later steps.

The digest is pinned rather than read from the GitHub release metadata, because
the releases API is rate limited per IP and made `install-bazel` fail on
unauthenticated runs. When bumping $bazeliskVersion, update the digest as well;
it is printed by

  gh release view <version> -R bazelbuild/bazelisk `
    --json assets --jq '.assets[] | "\(.name) \(.digest)"'

The installation directory is appended to `GITHUB_PATH` under GitHub Actions. On
Azure Pipelines the version and the executable path are published as pipeline
variables instead, so that the jobs do not have to pin the version a second time
in their own `variables:` block. Without either, the current process `PATH` is
updated, which makes the script convenient for local CI reproduction.
#>

$ErrorActionPreference = "Stop"

$bazeliskVersion = "v1.29.0"
$bazeliskSha256 = "092a8738d5b41aae7a85c42cc961b1034e3389aba43ffc20c0fabda7b43e095b"
$assetName = "bazelisk-windows-amd64.exe"
$installDir = Join-Path (Get-Location) "bazel\bin"
$bazelPath = Join-Path $installDir "bazel.exe"

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Invoke-WebRequest `
    -Uri "https://github.com/bazelbuild/bazelisk/releases/download/$bazeliskVersion/$assetName" `
    -OutFile $bazelPath

$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bazelPath).Hash.ToLowerInvariant()
if ($actualHash -ne $bazeliskSha256.ToLowerInvariant()) {
    Remove-Item -LiteralPath $bazelPath -Force
    throw ("Bazelisk SHA256 mismatch. Expected $bazeliskSha256, got $actualHash. " +
           "If you bumped `$bazeliskVersion, update the digest in this script.")
}

if ($env:GITHUB_PATH) {
    $installDir | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
}
elseif ($env:TF_BUILD) {
    Write-Host "##vso[task.setvariable variable=BAZELISK_VERSION]$bazeliskVersion"
    Write-Host "##vso[task.setvariable variable=BAZELISK_EXE]$bazelPath"
}
else {
    $env:PATH = "$installDir;$env:PATH"
}

& $bazelPath --version
