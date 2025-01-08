@echo off
rem ============================================================================
rem Copyright 2020 Intel Corporation
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem ============================================================================

rem req: PowerShell 3.0+
powershell.exe -command "if ($PSVersionTable.PSVersion.Major -ge 3) {exit 1} else {Write-Host \"The script requires PowerShell 3.0 or above (current version: $($PSVersionTable.PSVersion.Major).$($PSVersionTable.PSVersion.Minor))\"}" && goto Error_load

set TBBURLROOT=https://github.com/uxlfoundation/oneTBB/releases/download/v2021.5.0/
set TBBVERSION=oneapi-tbb-2021.5.0
set TBBPACKAGE=%TBBVERSION%-win

set TBBURL=%TBBURLROOT%%TBBPACKAGE%.zip
if /i "%1"=="" (
    set DST=%~dp0..\..\__deps\tbb
) else (
    set DST=%1\..\..\__deps\tbb
)

if not exist %DST% powershell.exe -command "New-Item -Path \"%DST%\" -ItemType Directory"
if not exist %DST%\win powershell.exe -command "New-Item -Path \"%DST%\win\" -ItemType Directory"
if not exist %DST%\win\tbb powershell.exe -command "New-Item -Path \"%DST%\win\tbb\" -ItemType Directory"

if not exist "%DST%\win\bin" (
    powershell.exe -command "(New-Object System.Net.WebClient).DownloadFile('%TBBURL%', '%DST%\%TBBPACKAGE%.zip')" && goto Unpack || goto Error_load

:Unpack
    powershell.exe -command "if (Get-Command Add-Type -errorAction SilentlyContinue) {Add-Type -Assembly \"System.IO.Compression.FileSystem\"; try { [IO.Compression.zipfile]::ExtractToDirectory(\"%DST%\%TBBPACKAGE%.zip\", \"%DST%\") ; Copy-Item \"%DST%\%TBBVERSION%\*\" -Destination \"%DST%\win\tbb\" -Recurse }catch{$_.exception ; exit 1}} else {exit 1}" || goto Error_unpack

    if not exist %DST%\win\tbb\redist\intel64\vc14 powershell.exe -command "New-Item -Path \"%DST%\win\tbb\redist\intel64\vc14\" -ItemType Directory"

    goto Exit 

:Error_load
    echo tbb.bat : Error: Failed to load %TBBURL% to %DST%, try to load it manually
    exit /B 1

:Error_unpack
    echo tbb.bat : Error: Failed to unpack %DST%\%TBBPACKAGE%.zip to %DST%, try unpack the archive manually
    exit /B 1

:Exit
    echo Downloaded and unpacked oneTBB small libraries to %DST%
    exit /B 0
) else (
    echo oneTBB small libraries are already installed in %DST%
    exit /B 0
)