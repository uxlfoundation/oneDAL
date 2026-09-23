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
Installs an upstream LLVM release and puts its `bin` directory on `PATH`.

.DESCRIPTION
`clang-cl` / `lld-link` / `llvm-lib` are the Windows ARM64 toolchain in both
build systems (dev/make/compiler_definitions/clang.ref.arm.mk and
dev/bazel/toolchains/cc_toolchain_win.bzl), and the runner images do not ship
them for that architecture.

Under GitHub Actions the directory is appended to `GITHUB_PATH` for subsequent
steps; otherwise only the current process `PATH` is updated, which keeps the
script usable for local CI reproduction.
#>

[CmdletBinding()]
param(
    [string] $Version = $env:LLVM_VERSION,
    [string] $Arch = $(if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE })
)

$ErrorActionPreference = "Stop"

if (-not $Version) {
    throw "LLVM version is not set; pass -Version or set LLVM_VERSION"
}

# LLVM names its Windows assets after the target, not after the arch id used by
# the rest of the CI scripts: ARM64 is `woa64` ("Windows on ARM 64").
$assetArch = switch ($Arch.ToUpperInvariant()) {
    "ARM64" { "woa64" }
    "AMD64" { "win64" }
    default { throw "Unsupported Windows host architecture for LLVM: $Arch" }
}

$installerName = "LLVM-$Version-$assetArch.exe"
Invoke-WebRequest "https://github.com/llvm/llvm-project/releases/download/llvmorg-$Version/$installerName" `
    -UseBasicParsing -OutFile $installerName
Start-Process -FilePath ".\$installerName" -ArgumentList "/S" -Wait

$binDir = Join-Path $env:ProgramFiles "LLVM\bin"
if ($env:GITHUB_PATH) {
    $binDir | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
}
else {
    $env:PATH = "$binDir;$env:PATH"
}
