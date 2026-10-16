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
This is the PowerShell counterpart of `.ci/env/bazelisk.sh`. The Windows
nightly job cannot use that POSIX-shell helper, so this script downloads the
Windows asset, verifies it against the SHA256 digest pinned below, and
makes `bazel.exe` available to later workflow steps.

When GitHub Actions provides `GITHUB_PATH`, the installation directory is
appended there for subsequent steps. Otherwise, the current process `PATH` is
updated, which also makes the script convenient for local CI reproduction.
#>

$ErrorActionPreference = "Stop"

$bazeliskVersion = "v1.29.0"
$hostArch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
$assetSuffix = switch ($hostArch.ToUpperInvariant()) {
    "ARM64" { "arm64" }
    "AMD64" { "amd64" }
    default { throw "Unsupported Windows host architecture for Bazelisk: $hostArch" }
}
$assetName = "bazelisk-windows-$assetSuffix.exe"
$installDir = Join-Path (Get-Location) "bazel\bin"
$bazelPath = Join-Path $installDir "bazel.exe"

# Release asset digests, pinned so the install makes no rate-limited GitHub API
# call. Add the digests for every new $bazeliskVersion:
#   gh api repos/bazelbuild/bazelisk/releases/tags/<version> --jq '.assets[] | "\(.name) \(.digest)"'
$pinnedDigests = @{
    "v1.29.0/bazelisk-windows-amd64.exe" = "092a8738d5b41aae7a85c42cc961b1034e3389aba43ffc20c0fabda7b43e095b"
    "v1.29.0/bazelisk-windows-arm64.exe" = "8bc42bd5d7857f18a21440b906469bb6c7cf91a7c72364d4b1e5ec56a76fe94f"
}
$expectedHash = $pinnedDigests["$bazeliskVersion/$assetName"]
if (-not $expectedHash) {
    throw ("No pinned SHA256 for $assetName at $bazeliskVersion in .ci/env/bazelisk.ps1. Add it from: " +
           "gh api repos/bazelbuild/bazelisk/releases/tags/$bazeliskVersion --jq '.assets[] | `"\(.name) \(.digest)`"'")
}

# Release downloads fail transiently in CI (dropped connections, 5xx), so retry
# them with exponential backoff, as `.ci/env/bazelisk.sh` does.
$fetchAttempts = if ($env:BAZELISK_FETCH_ATTEMPTS) { [int]$env:BAZELISK_FETCH_ATTEMPTS } else { 5 }
$fetchDelay = if ($env:BAZELISK_FETCH_DELAY) { [int]$env:BAZELISK_FETCH_DELAY } else { 5 }
if ($fetchAttempts -lt 1 -or $fetchDelay -lt 0) {
    throw "BAZELISK_FETCH_ATTEMPTS must be at least 1 and BAZELISK_FETCH_DELAY non-negative"
}

$downloadUrl = "https://github.com/bazelbuild/bazelisk/releases/download/$bazeliskVersion/$assetName"
New-Item -ItemType Directory -Force -Path $installDir | Out-Null
$delay = $fetchDelay
for ($attempt = 1; ; $attempt++) {
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $bazelPath
        break
    }
    catch {
        if ($attempt -ge $fetchAttempts) {
            throw "Giving up after $attempt attempts to download $assetName. Last error: $($_.Exception.Message)"
        }
        Write-Host "Attempt $attempt of $fetchAttempts to download $assetName failed: $($_.Exception.Message)"
        Write-Host "Retrying in $delay s"
        Start-Sleep -Seconds $delay
        $delay = $delay * 2
    }
}

$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bazelPath).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash.ToLowerInvariant()) {
    # Do not leave an unverified executable where the caller would add it to PATH.
    Remove-Item -LiteralPath $bazelPath -Force -ErrorAction SilentlyContinue
    throw "Bazelisk SHA256 mismatch. Expected $expectedHash, got $actualHash"
}

$pathLine = $installDir
if ($env:GITHUB_PATH) {
    $pathLine | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
}
else {
    $env:PATH = "$installDir;$env:PATH"
}

& $bazelPath --version
