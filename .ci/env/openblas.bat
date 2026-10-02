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
rem
rem `INTERFACE64=ON` builds the ILP64 interface -- BLAS/LAPACK integer arguments
rem are 64-bit -- as the Linux script does by default (`.ci/env/openblas.sh
rem --ilp64 on` -> `INTERFACE64=1`). This is not a preference but a requirement of
rem how oneDAL calls the reference backend: `DAAL_INT` is `__int64` on
rem `_WIN64`/`TARGET_ARM` (`cpp/daal/include/services/daal_defines.h:78-88`) and
rem the wrappers in `cpp/daal/src/externals/service_blas_ref.h` hand
rem `const DAAL_INT *` straight to `dgemm_`/`dsyrk_`/... with no narrowing cast.
rem An LP64 OpenBLAS exports those same names taking 32-bit integers, so the
rem mismatch does not break the link: the callee reads half of each argument it
rem is passed, and the failure surfaces as wrong results or an out-of-bounds
rem access instead.
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
        -DCMAKE_SYSTEM_PROCESSOR=arm64 ^
        -DCMAKE_SYSTEM_NAME=Windows ^
        -DCMAKE_INSTALL_PREFIX="%DST%"
    if errorlevel 1 (popd & goto Error_build)
    cmake --build build-arm64
    if errorlevel 1 (popd & goto Error_build)
    cmake --install build-arm64
    if errorlevel 1 (popd & goto Error_build)
popd

rem `INTERFACE64` also renames what gets installed: OpenBLAS's CMake build appends
rem `_64` to the library name (`OpenBLAS_LIBNAME`, CMakeLists.txt:128-133) and puts
rem the headers in `include/openblas64` (CMakeLists.txt:700). The Makefile build the
rem Linux script uses does neither, so the ILP64 Linux package stays plain
rem `libopenblas.a` + `include/`, and that is the layout both consumers expect on
rem Windows too (`releaseopen_blas.LIBS_A` in dev/make/deps.ref.mk,
rem `dev/bazel/deps/openblas.bzl`). Only file names change -- `SYMBOLSUFFIX` is left
rem empty, so the exported symbols are still `dgemm_`/`dsyrk_`/... -- and the build
rem is static, so there is no import library pointing at a DLL whose name would also
rem have to change. Renaming here keeps the interface change out of every consumer.
if exist "%DST%\lib\openblas_64.lib" (
    copy /Y "%DST%\lib\openblas_64.lib" "%DST%\lib\openblas.lib" >nul
    if errorlevel 1 goto Error_layout
)
if exist "%DST%\include\openblas64\openblas_config.h" (
    xcopy /E /I /Y "%DST%\include\openblas64" "%DST%\include" >nul
    if errorlevel 1 goto Error_layout
)

rem The interface width is what oneDAL silently depends on and nothing downstream
rem can detect: the symbol names are identical either way, so an LP64 package links
rem and then misbehaves. Read it back off the installed configuration header, which
rem the CMake build fills from `config.h` (CMakeLists.txt:705-712) and therefore
rem carries `#define OPENBLAS_USE64BITINT` only for an ILP64 build. A renamed or
rem dropped CMake option is then caught here rather than in numerical results.
rem
rem The `#define` has to be part of the pattern: `openblas_config_template.h` is
rem appended to every generated header and contains `#ifdef OPENBLAS_USE64BITINT`
rem unconditionally, so matching the bare name passes for an LP64 build too.
rem Measured on 0.3.34 by configuring both ways and counting matches in the
rem generated header: bare name 1 vs 1, `#define OPENBLAS_USE64BITINT` 0 vs 1.
findstr /C:"#define OPENBLAS_USE64BITINT" "%DST%\include\openblas_config.h" >nul 2>&1
if errorlevel 1 (
    echo openblas.bat : Error: the installed OpenBLAS is not ILP64 -- "#define OPENBLAS_USE64BITINT" is absent from "%DST%\include\openblas_config.h", while oneDAL passes 64-bit DAAL_INT arguments to it
    exit /B 1
)
if not exist "%DST%\lib\openblas.lib" (
    echo openblas.bat : Error: "%DST%\lib\openblas.lib" was not produced; oneDAL links OpenBLAS under that name
    exit /B 1
)

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
