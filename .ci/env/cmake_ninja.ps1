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
Installs arch-matched CMake and Ninja.

.DESCRIPTION
Both are needed to build the OpenBLAS and oneTBB dependencies from source
(.ci/env/openblas.bat, .ci/env/tbb.bat), and the ARM64 runner images either
lack them or provide x86_64 builds.

Under GitHub Actions the CMake directory is appended to `GITHUB_PATH` for
subsequent steps; otherwise only the current process `PATH` is updated. Ninja is
a single executable and is copied into a directory that is already on `PATH`.
#>

[CmdletBinding()]
param(
    [string] $CMakeVersion = $env:CMAKE_VERSION,
    [string] $NinjaVersion = $env:NINJA_VERSION,
    [string] $Arch = $(if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE })
)

$ErrorActionPreference = "Stop"

if (-not $CMakeVersion) {
    throw "CMake version is not set; pass -CMakeVersion or set CMAKE_VERSION"
}
if (-not $NinjaVersion) {
    throw "Ninja version is not set; pass -NinjaVersion or set NINJA_VERSION"
}

$archId = $Arch.ToLowerInvariant()
if ($archId -notin @("arm64", "amd64")) {
    throw "Unsupported Windows host architecture for CMake/Ninja: $Arch"
}

$msiName = "cmake-$CMakeVersion-windows-$archId.msi"
Invoke-WebRequest "https://github.com/Kitware/CMake/releases/download/v$CMakeVersion/$msiName" -OutFile $msiName
Start-Process msiexec.exe -ArgumentList "/i $msiName /quiet /norestart" -Wait

$cmakeBin = Join-Path $env:ProgramFiles "CMake\bin"
if ($env:GITHUB_PATH) {
    $cmakeBin | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
}
else {
    $env:PATH = "$cmakeBin;$env:PATH"
}

# Ninja spells the ARM64 asset `ninja-winarm64.zip` and the x86_64 one
# `ninja-win.zip`.
$ninjaAsset = if ($archId -eq "arm64") { "ninja-winarm64.zip" } else { "ninja-win.zip" }
Invoke-WebRequest "https://github.com/ninja-build/ninja/releases/download/v$NinjaVersion/$ninjaAsset" -OutFile $ninjaAsset
Expand-Archive $ninjaAsset -DestinationPath ninja -Force
Copy-Item ninja\ninja.exe -Destination "C:\Windows\System32"
