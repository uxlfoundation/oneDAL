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
Windows asset, verifies the SHA256 digest published in the GitHub release, and
makes `bazel.exe` available to later workflow steps.

When GitHub Actions provides `GITHUB_PATH`, the installation directory is
appended there for subsequent steps. Otherwise, the current process `PATH` is
updated, which also makes the script convenient for local CI reproduction.
#>

$ErrorActionPreference = "Stop"

$bazeliskVersion = "v1.29.0"
$assetName = "bazelisk-windows-amd64.exe"
$installDir = Join-Path (Get-Location) "bazel\bin"
$bazelPath = Join-Path $installDir "bazel.exe"

$headers = @{
    Accept = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

if ($env:GITHUB_TOKEN) {
    # Authentication is optional, but avoids GitHub API rate limits in CI.
    $headers.Authorization = "Bearer $env:GITHUB_TOKEN"
}

# Both the GitHub API and the release CDN fail transiently in CI: the API
# rate-limits unauthenticated callers per runner IP, and either endpoint can
# drop a connection. One failure used to take the whole job down, so retry with
# exponential backoff, as `.ci/env/bazelisk.sh` does.
$fetchAttempts = if ($env:BAZELISK_FETCH_ATTEMPTS) { [int]$env:BAZELISK_FETCH_ATTEMPTS } else { 5 }
$fetchDelay = if ($env:BAZELISK_FETCH_DELAY) { [int]$env:BAZELISK_FETCH_DELAY } else { 5 }

function Invoke-WithRetry {
    param(
        [Parameter(Mandatory = $true)][scriptblock] $Action,
        [Parameter(Mandatory = $true)][string] $Description
    )

    $delay = $fetchDelay
    for ($attempt = 1; $attempt -le $fetchAttempts; $attempt++) {
        try {
            return & $Action
        }
        catch {
            if ($attempt -ge $fetchAttempts) {
                throw "Giving up after $attempt attempts: $Description. Last error: $($_.Exception.Message)"
            }
            Write-Host "Attempt $attempt of $fetchAttempts failed ($Description): $($_.Exception.Message)"
            Write-Host "Retrying in $delay s"
            Start-Sleep -Seconds $delay
            $delay = $delay * 2
        }
    }
}

$release = Invoke-WithRetry -Description "fetch Bazelisk $bazeliskVersion release metadata" -Action {
    Invoke-RestMethod `
        -Headers $headers `
        -Uri "https://api.github.com/repos/bazelbuild/bazelisk/releases/tags/$bazeliskVersion"
}

$asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
if (-not $asset) {
    throw "Could not find $assetName in Bazelisk release $bazeliskVersion"
}

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Invoke-WithRetry -Description "download $assetName" -Action {
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $bazelPath
} | Out-Null

if (-not $asset.digest.StartsWith("sha256:")) {
    # Do not install an asset that cannot be checked against release metadata.
    throw "Unexpected Bazelisk digest format: $($asset.digest)"
}

$expectedHash = $asset.digest.Substring("sha256:".Length)
$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bazelPath).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash.ToLowerInvariant()) {
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
