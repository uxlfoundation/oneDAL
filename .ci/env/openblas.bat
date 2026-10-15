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
set BLASVERSION=0.3.34
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

rem Installed into from scratch: the checks below read what is in %DST%, and a
rem stale library from an earlier local run would pass them.
if exist "%DST%" rmdir /s /q "%DST%"
mkdir "%DST%" >nul
if not exist "%BLASSOURCEDIR%" mkdir "%BLASSOURCEDIR%" >nul

curl -L -o "%BLASSOURCEDIR%\openblas.zip" "%BLASURL%"
if errorlevel 1 goto Error_load

tar -xf "%BLASSOURCEDIR%\openblas.zip" -C "%BLASSOURCEDIR%"
if errorlevel 1 goto Error_unpack

rem Static, like the Linux build (.ci/env/openblas.sh): dev/make/deps.ref.mk links
rem `openblas.$(a)` into onedal_core, so the BLAS/LAPACK symbols travel inside
rem oneDAL's own binaries. A shared build would leave `onedal_core.<major>.dll`
rem importing a module named `openblas.dll`, and Windows resolves imports by base
rem name, so any other `openblas.dll` already in the process would satisfy it.
rem
rem `NOFORTRAN` + `C_LAPACK`: LAPACK from the f2c-translated sources, as the Linux
rem script gets from `NO_FORTRAN=1`. Fortran objects would add
rem `/DEFAULTLIB:flang_rt.runtime.dynamic` to the archive and every consumer would
rem have to find the flang runtime.
rem
rem `USE_THREAD=OFF` + `USE_LOCKING=ON`, as on Linux: oneDAL parallelises through
rem oneTBB, and both `onedal_core` and `onedal_thread` embed this archive, so a
rem threaded OpenBLAS would add pools of its own. Locking keeps the
rem single-threaded library callable from several TBB threads.
rem
rem `INTERFACE64=ON`: oneDAL's `DAAL_INT` is `__int64` here
rem (cpp/daal/include/services/daal_defines.h) and the wrappers in
rem `cpp/daal/src/externals/service_blas_ref.h` pass `const DAAL_INT *` straight to
rem `dgemm_`/`dsyrk_`/..., so the interface has to be ILP64. The symbol names are
rem identical either way, so an LP64 build links and then misbehaves.
rem
rem No `CMAKE_SYSTEM_NAME`/`CMAKE_SYSTEM_PROCESSOR`: the runner is native ARM64
rem (enforced above), and setting them makes CMake treat this as cross-compiling,
rem which is the one case where `INTERFACE64` does not reach the library. OpenBLAS
rem then writes `config.h` itself without `USE64BITINT` (cmake/prebuild.cmake),
rem because that macro is otherwise passed only to `getarch` (cmake/system.cmake)
rem and emitted by `getarch_2nd`, while the compile-side line stays commented out.
rem `SUFFIX64` still renames the library to `openblas_64`, so the result is an LP64
rem library with an ILP64 name -- measured locally: `sizeof(blasint)` is 4 with
rem those two options and 8 without them.
pushd "%BLASSOURCEDIR%\OpenBLAS-%BLASVERSION%"
    if exist build-arm64 rmdir /s /q build-arm64
    cmake -B build-arm64 -S . -GNinja ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DTARGET=ARMV8 ^
        -DBINARY=64 ^
        -DINTERFACE64=ON ^
        -DCMAKE_C_COMPILER=clang-cl ^
        -DCMAKE_CXX_COMPILER=clang-cl ^
        -DNOFORTRAN=ON ^
        -DC_LAPACK=ON ^
        -DUSE_THREAD=OFF ^
        -DUSE_LOCKING=ON ^
        -DBUILD_SHARED_LIBS=OFF ^
        -DCMAKE_INSTALL_PREFIX="%DST%"
    if errorlevel 1 (popd & goto Error_build)
    cmake --build build-arm64
    if errorlevel 1 (popd & goto Error_build)
    cmake --install build-arm64
    if errorlevel 1 (popd & goto Error_build)
popd

rem The ILP64 witness is the archive, not the file name: `SUFFIX64` renames the
rem library from `INTERFACE64` alone, so an LP64 build also installs
rem `openblas_64.lib`. ` USE64BITINT ` is compiled into `openblas_get_config()`
rem (driver/others/openblas_get_config.c) only when the library really is ILP64.
rem findstr's exit code is read explicitly, as `>nul` plus `errorlevel` rather than
rem through a pipe, because it reports a binary match on stdout.
findstr /C:" USE64BITINT " "%DST%\lib\openblas_64.lib" >nul 2>&1
if errorlevel 1 (
    echo openblas.bat : Error: the installed OpenBLAS is not ILP64 -- "openblas_get_config" in "%DST%\lib\openblas_64.lib" does not report USE64BITINT, while oneDAL passes 64-bit DAAL_INT arguments to it
    exit /B 1
)

rem `INTERFACE64` also renames what is installed: the CMake build appends `_64` to
rem the library name and nests the headers in `include/openblas64`. The Makefile
rem build the Linux script uses does neither, so both consumers
rem (`releaseopen_blas.LIBS_A` in dev/make/deps.ref.mk, dev/bazel/deps/openblas.bzl)
rem expect plain `openblas.lib`. Only names change -- `SYMBOLSUFFIX` is empty, so the
rem exported symbols stay `dgemm_`/`dsyrk_`/... -- and nothing includes the headers,
rem so they are left where CMake put them.
copy /Y "%DST%\lib\openblas_64.lib" "%DST%\lib\openblas.lib" >nul
if errorlevel 1 goto Error_layout

echo Downloaded and unpacked OpenBlas small libraries to %DST%
exit /B 0

:Error_load
    echo openblas.bat : Error: Failed to load %BLASURL% to %BLASSOURCEDIR%, try to load it manually
    exit /B 1

:Error_unpack
    echo openblas.bat : Error: Failed to unpack %BLASSOURCEDIR%\openblas.zip to %BLASSOURCEDIR%, try unpack the archive manually
    exit /B 1

:Error_layout
    echo openblas.bat : Error: Failed to normalize the installed OpenBLAS layout under %DST%
    exit /B 1

:Error_build
    echo openblas.bat : Error: Failed to configure, build or install OpenBLAS from %BLASSOURCEDIR%\OpenBLAS-%BLASVERSION% into %DST%
    exit /B 1
