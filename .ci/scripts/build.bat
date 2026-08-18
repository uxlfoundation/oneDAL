@echo off
rem ============================================================================
rem Copyright 2022 Intel Corporation
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

rem %1 - Make target
rem %2 - Compiler
rem %3 - Instruction set

set errorcode=0
echo CPUCOUNT=%NUMBER_OF_PROCESSORS%

echo "PATH=C:\Program Files\LLVM\bin;C:\msys64\usr\bin;%PATH%"
set "PATH=C:\Program Files\LLVM\bin;C:\msys64\usr\bin;%PATH%"

echo pacman -S --noconfirm msys/make
pacman -S --noconfirm msys/make

IF "%VS_VER%"=="2017_build_tools" (
    @call "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
    echo "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
) ELSE IF "%VS_VER%"=="2019_build_tools" (
    @call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
    echo "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
) ELSE (
    @call "%ONEAPI_ROOT%\setvars-vcvarsall.bat" %VS_VER%
    echo "%ONEAPI_ROOT%\setvars-vcvarsall.bat" %VS_VER%
)

set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"

if /I "%ARCH%"=="AMD64" (
    set "PLAT=win32e"
    set "ARCH_DIR=intel64"
) else if /I "%ARCH%"=="ARM64" (
    set "PLAT=winarm"
    set "ARCH_DIR=ARM64"
) else (
    echo Unknown architecture: %ARCH%
    exit /b 1
)

echo make %1 -j%NUMBER_OF_PROCESSORS% COMPILER=%2 PLAT="%PLAT%" REQCPU=%3
make %1 -j%NUMBER_OF_PROCESSORS% COMPILER=%2 PLAT="%PLAT%" REQCPU=%3 || set errorcode=1

cmake -DINSTALL_DIR=__release_win_%2\daal\latest\lib\cmake\oneDAL -DARCH_DIR="%ARCH_DIR%" -P cmake\scripts\generate_config.cmake || set errorcode=1
rem No cmake config generation here. This script used to call
rem cmake/scripts/generate_config.cmake itself, added by PR #2222 (merged
rem 2023-01-04). Two weeks later PR #2243 (merged 2023-01-20) taught the makefile
rem to stage the oneDALConfig files on its own through `_release_cmake_configs`,
rem which made the call here redundant. Both PR numbers are from January 2023 --
rem this code has been dead for over three years.
rem
rem Keeping it hurt: the second `configure_file` pass overwrote the staged files
rem with output that did not reproduce them byte for byte, which the release
rem comparator reported as a text mismatch. `.ci/scripts/build.sh` has never had
rem an equivalent call, which is why Linux compared clean.
EXIT /B %errorcode%
