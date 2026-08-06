@echo off
rem ============================================================================
rem Copyright contributors to the oneDAL project
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

IF "%VS_VER%"=="2026_build_tools" (
    @call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
) ELSE IF "%VS_VER%"=="2019_build_tools" (
    @call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
) ELSE IF "%VS_VER%"=="2017_build_tools" (
    @call "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
)

if not exist "%DST%" mkdir "%DST%" >nul
if not exist "%BLASSOURCEDIR%" mkdir "%BLASSOURCEDIR%" >nul

curl -L -o "%BLASSOURCEDIR%\openblas.zip" "%BLASURL%"
if errorlevel 1 goto Error_load

tar -xf "%BLASSOURCEDIR%\openblas.zip" -C "%BLASSOURCEDIR%"
if errorlevel 1 goto Error_unpack

pushd "%BLASSOURCEDIR%\OpenBLAS-%BLASVERSION%"
    if exist build-arm64 rmdir /s /q build-arm64
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