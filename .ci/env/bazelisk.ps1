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
    $headers.Authorization = "Bearer $env:GITHUB_TOKEN"
}

$release = Invoke-RestMethod `
    -Headers $headers `
    -Uri "https://api.github.com/repos/bazelbuild/bazelisk/releases/tags/$bazeliskVersion"

$asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
if (-not $asset) {
    throw "Could not find $assetName in Bazelisk release $bazeliskVersion"
}

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $bazelPath

if (-not $asset.digest.StartsWith("sha256:")) {
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
