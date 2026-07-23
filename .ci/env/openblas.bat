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

setlocal

rem req: PowerShell 3.0+
powershell.exe -command "if ($PSVersionTable.PSVersion.Major -lt 3) {Write-Host \"The script requires PowerShell 3.0 or above (current version: $($PSVersionTable.PSVersion.Major).$($PSVersionTable.PSVersion.Minor))\"; exit 1} else {exit 0}"
if errorlevel 1 goto Error_load

if /I not "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
    echo Current only available for ARM64.
    exit /B 1
)

if /i "%1"=="" (
    set DST=%~dp0..\..\__deps\open_blas
) else (
    set DST=%1\..\..\__deps\open_blas
)
set BLASSOURCEDIR=%~dp0..\..\__work\openblas
set BLASVERSION=0.3.33
set BLASURLROOT=https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v%BLASVERSION%
set BLASPACKAGE=
set BLASURL=%BLASURLROOT%%BLASPACKAGE%.zip
set "PATH=C:\Program Files\LLVM\bin;%PATH%"

if not exist "%DST%" powershell.exe -command "New-Item -Path \"%DST%\" -ItemType Directory" >nul
if not exist "%BLASSOURCEDIR%" powershell.exe -command "New-Item -Path \"%BLASSOURCEDIR%\" -ItemType Directory" >nul

powershell.exe -command "(New-Object System.Net.WebClient).DownloadFile('%BLASURL%', '%BLASSOURCEDIR%\openblas.zip')"
if errorlevel 1 goto Error_load

powershell.exe -command "if (Get-Command Add-Type -errorAction SilentlyContinue) {Add-Type -Assembly \"System.IO.Compression.FileSystem\"; try { [IO.Compression.zipfile]::ExtractToDirectory(\"%BLASSOURCEDIR%\openblas.zip\", \"%BLASSOURCEDIR%\") } catch { $_.exception; exit 1 }} else {exit 1}"
if errorlevel 1 goto Error_unpack

pushd "%BLASSOURCEDIR%\OpenBlas-%BLASVERSION%"
    rmdir /s /q build-arm64
    cmake -B build-arm64 -S . -GNinja ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DTARGET=ARMV8 ^
        -DBINARY=64 ^
        -DCMAKE_C_COMPILER=clang-cl ^
        -DCMAKE_CXX_COMPILER=clang-cl ^
        -DCMAKE_Fortran_COMPILER=flang-new ^
        -DBUILD_SHARED_LIBS=ON ^
        -DCMAKE_SYSTEM_PROCESSOR=arm64 ^
        -DCMAKE_SYSTEM_NAME=Windows ^
        -DCMAKE_INSTALL_PREFIX="%DST%"
    cmake --build build-arm64
    cmake --install build-arm64
popd

echo Downloaded and unpacked OpenBlas small libraries to %DST%
exit /B 0

:Error_load
    echo openblas.bat : Error: Failed to load %BLASURL% to %BLASSOURCEDIR%, try to load it manually
    exit /B 1

:Error_unpack
    echo openblas.bat : Error: Failed to unpack %BLASSOURCEDIR%\openblas.zip to %BLASSOURCEDIR%, try unpack the archive manually
    exit /B 1