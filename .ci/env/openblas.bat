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
set "PATH=%ProgramFiles%\LLVM\bin;%PATH%"

IF "%VS_VER%"=="2026_build_tools" (
    @call "%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
) ELSE IF "%VS_VER%"=="2019_build_tools" (
    @call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
) ELSE IF "%VS_VER%"=="2017_build_tools" (
    @call "%ProgramFiles(x86)%\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
)

if not exist "%DST%" mkdir "%DST%" >nul
if not exist "%BLASSOURCEDIR%" mkdir "%BLASSOURCEDIR%" >nul

curl -L -o "%BLASSOURCEDIR%\openblas.zip" "%BLASURL%"
if errorlevel 1 goto Error_load

tar -xf "%BLASSOURCEDIR%\openblas.zip" -C "%BLASSOURCEDIR%"
if errorlevel 1 goto Error_unpack

rem Built static, like the Linux build (.ci/env/openblas.sh): dev/make/deps.ref.mk
rem links `openblas.$(a)` into onedal_core, so the BLAS/LAPACK symbols travel
rem inside oneDAL's own binaries. A shared build yields an import library
rem instead, which leaves `onedal_core.<major>.dll` importing a DLL named
rem `openblas.dll`; Windows resolves imports by base name against the modules
rem already loaded in the process, so any other wheel shipping its own
rem `openblas.dll` would satisfy that import.
rem
rem `NOFORTRAN` + `C_LAPACK` build LAPACK from the f2c-translated sources in
rem `lapack-netlib/SRC`, which is what the Linux script gets from `NO_FORTRAN=1`.
rem Fortran objects would otherwise put `/DEFAULTLIB:flang_rt.runtime.dynamic`
rem into the archive, and every consumer of `openblas.lib` -- oneDAL's own DLLs
rem first of all -- would then have to find the flang runtime at link time.
rem
rem `USE_THREAD=OFF` + `USE_LOCKING=ON`, again as on Linux: oneDAL parallelises
rem through oneTBB, so OpenBLAS must not bring a thread pool of its own. The
rem CMake build defaults `USE_THREAD` to 1 whenever the machine has two cores
rem (cmake/system.cmake), and with a static OpenBLAS both `onedal_core` and
rem `onedal_thread` embed the archive, so a threaded build would put two
rem independent pools in one process on top of TBB's. `USE_LOCKING` keeps the
rem single-threaded library safe to call from several TBB threads at once.
pushd "%BLASSOURCEDIR%\OpenBLAS-%BLASVERSION%"
    if exist build-arm64 rmdir /s /q build-arm64
    cmake -B build-arm64 -S . -GNinja ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DTARGET=ARMV8 ^
        -DBINARY=64 ^
        -DCMAKE_C_COMPILER=clang-cl ^
        -DCMAKE_CXX_COMPILER=clang-cl ^
        -DNOFORTRAN=ON ^
        -DC_LAPACK=ON ^
        -DUSE_THREAD=OFF ^
        -DUSE_LOCKING=ON ^
        -DBUILD_SHARED_LIBS=OFF ^
        -DCMAKE_SYSTEM_PROCESSOR=arm64 ^
        -DCMAKE_SYSTEM_NAME=Windows ^
        -DCMAKE_INSTALL_PREFIX="%DST%"
    if errorlevel 1 (popd & goto Error_build)
    cmake --build build-arm64
    if errorlevel 1 (popd & goto Error_build)
    cmake --install build-arm64
    if errorlevel 1 (popd & goto Error_build)
popd

echo Downloaded and unpacked OpenBlas small libraries to %DST%
exit /B 0

:Error_load
    echo openblas.bat : Error: Failed to load %BLASURL% to %BLASSOURCEDIR%, try to load it manually
    exit /B 1

:Error_unpack
    echo openblas.bat : Error: Failed to unpack %BLASSOURCEDIR%\openblas.zip to %BLASSOURCEDIR%, try unpack the archive manually
    exit /B 1

:Error_build
    echo openblas.bat : Error: Failed to configure, build or install OpenBLAS from %BLASSOURCEDIR%\OpenBLAS-%BLASVERSION% into %DST%
    exit /B 1
