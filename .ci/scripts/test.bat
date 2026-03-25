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

setlocal enabledelayedexpansion enableextensions

rem %1 - Examples target
rem %2 - Examples linking
rem %3 - Compiler
rem %4 - build system
set examples=%1
set linking=%2
set compiler=%3
set build_system=%4

set errorcode=0

if "%compiler%"=="icx" (
    set release_dir=__release_win
) else if "%compiler%"=="msvs" (
    set release_dir=__release_win_vc
) else (
    set release_dir=__release_win_%compiler%
)

for /f "tokens=*" %%i in ('python -c "from multiprocessing import cpu_count; print(cpu_count())"') do set CPUCOUNT=%%i
echo CPUCOUNT=%CPUCOUNT%

echo PATH=C:\msys64\usr\bin;%PATH%
set PATH=C:\msys64\usr\bin;%PATH%

echo "%VISUALSTUDIOVERSION% HERE"

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

echo call %release_dir%\daal\latest\env\vars.bat
call %release_dir%\daal\latest\env\vars.bat || set errorcode=1

echo set LIB=%~dp0..\..\%release_dir%\tbb\latest\lib\intel64\vc_mt;%LIB%
set LIB=%~dp0..\..\%release_dir%\tbb\latest\lib\intel64\vc_mt;%LIB%
echo set PATH=%~dp0..\..\%release_dir%\tbb\latest\lib\intel64\vc_mt;%PATH%
set PATH=%~dp0..\..\%release_dir%\tbb\latest\lib\intel64\vc_mt;%PATH%

echo set LIB=%~dp0..\..\%release_dir%\tbb\latest\redist\intel64\vc_mt;%LIB%
set LIB=%~dp0..\..\%release_dir%\tbb\latest\redist\intel64\vc_mt;%LIB%
echo set PATH=%~dp0..\..\%release_dir%\tbb\latest\redist\intel64\vc_mt;%PATH%
set PATH=%~dp0..\..\%release_dir%\tbb\latest\redist\intel64\vc_mt;%PATH%

echo set TBB_DIR=%~dp0..\..\__deps\tbb\win\tbb\lib\cmake\tbb
set TBB_DIR=%~dp0..\..\__deps\tbb\win\tbb\lib\cmake\tbb

echo %release_dir%\daal\latest\examples\%examples%
cd %release_dir%\daal\latest\examples\%examples%

set cmake_link_mode_short=so
set cmake_link_mode=dynamic
if "%link_mode%"=="lib" (
    set cmake_link_mode_short=a
    set cmake_link_mode=static
)

if "%build_system%"=="cmake" (
    if exist Build rd /S /Q Build
    md Build

    set results_dir=_cmake_results\intel_intel64_%cmake_link_mode_short%\Release
    echo cmake -B Build -S . -DONEDAL_LINK=%cmake_link_mode% -DTBB_DIR=%TBB_DIR%
    cmake -B Build -S . -DONEDAL_LINK=%cmake_link_mode% -DTBB_DIR=%TBB_DIR% || set errorcode=1
    set solution_name=%examples:\=_%
    msbuild.exe "Build\!solution_name!_examples.sln" /p:Configuration=Release || set errorcode=1

    for /f "delims=." %%F in ('dir /B !results_dir!\*.exe 2^> nul') do (
        set example=%%F
        echo !example! >> cmake_examples_list.txt
    )

    for /f %%G in (cmake_examples_list.txt) do (
        set ExampleName=%%G
        set exe_log=%%G.res
        if exist !results_dir!\%%G.exe (
            !results_dir!\%%G.exe 2>&1 > !exe_log!
            set errorcode=!errorlevel!
            if !errorcode! EQU 0 (
                set status_ex=PASSED !ExampleName!
                echo !status_ex!
            ) else (
                echo FAILED !ExampleName! with errno !errorcode!
            )
        )
    )

    echo xcopy *.res "!results_dir!" /I /H /Q /R /Y
    xcopy *.res "!results_dir!" /I /H /Q /R /Y

) else (
    if "%examples%"=="daal\cpp" nmake %linking% compiler=%compiler%
    if "%examples%"=="oneapi\cpp" nmake %linking% compiler=%compiler%
)
EXIT /B %errorcode%
