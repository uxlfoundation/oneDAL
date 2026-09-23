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
Installs MSYS2 into `C:\msys64`, replacing whatever the runner image ships.

.DESCRIPTION
The Make and Bazel Windows lanes both need a Bourne shell (Bazel additionally
requires one for its shell-based actions), and the ARM64 runner images carry an
x86_64 MSYS2 at best. Any pre-existing installation is removed first so the
arch-matched one is what ends up on disk.
#>

[CmdletBinding()]
param(
    [string] $Arch = $(if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }),
    [string] $Version = "2026-06-11"
)

$ErrorActionPreference = "Stop"

$archId = $Arch.ToLowerInvariant()
# The installer file name spells the date without separators.
$installerName = "msys2-$archId-$($Version -replace '-', '').exe"

if (Test-Path 'C:\msys64') {
    Remove-Item 'C:\msys64' -Recurse -Force
}

Invoke-WebRequest "https://github.com/msys2/msys2-installer/releases/download/$Version/$installerName" `
    -UseBasicParsing -OutFile $installerName
& ".\$installerName" install --confirm-command --root "C:\msys64"
