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

echo PATH=C:\msys64\usr\bin;%PATH%
set PATH=C:\msys64\usr\bin;%PATH%

echo pacman -S --noconfirm msys/make
pacman -S --noconfirm msys/make

IF "%VS_VER%"=="2017_build_tools" (
    @call "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    echo "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) ELSE (
    IF "%VS_VER%"=="2019_build_tools" (
        @call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        echo "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) ELSE (
          @call "%ONEAPI_ROOT%\setvars-vcvarsall.bat" %VS_VER%
          echo "%ONEAPI_ROOT%\setvars-vcvarsall.bat" %VS_VER%
    )
)

echo make %1 -j%NUMBER_OF_PROCESSORS% COMPILER=%2 PLAT=win32e REQCPU=%3
make %1 -j%NUMBER_OF_PROCESSORS% COMPILER=%2 PLAT=win32e REQCPU=%3 || set errorcode=1

cmake -DINSTALL_DIR=__release_win_vc\daal\latest\lib\cmake\oneDAL -DARCH_DIR=intel64 -P cmake\scripts\generate_config.cmake || set errorcode=1
EXIT /B %errorcode%
