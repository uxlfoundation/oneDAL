@echo off
rem ============================================================================
rem Copyright 2020 Intel Corporation
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

rem Pure CMD version with PowerShell fallback for extraction

setlocal enabledelayedexpansion

set TBBVERSION=2023.0.0
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
    set TBBURLROOT=https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v%TBBVERSION%
    set TBBPACKAGE=
) else (
    set TBBURLROOT=https://github.com/uxlfoundation/oneTBB/releases/download/v%TBBVERSION%/
    set TBBPACKAGE=oneapi-tbb-%TBBVERSION%-win
)

set TBBURL=%TBBURLROOT%%TBBPACKAGE%.zip

if /i "%1"=="" (
    set "DST=%~dp0..\..\__deps\tbb"
) else (
    set "DST=%1\..\..\__deps\tbb"
)

rem Create directories
if not exist "%DST%" mkdir "%DST%"
if not exist "%DST%\win" mkdir "%DST%\win"
if not exist "%DST%\win\tbb" mkdir "%DST%\win\tbb"

if not exist "%DST%\win\bin" (
    rem Download TBB archive
    echo Downloading %TBBURL% ...
    certutil -urlcache -split -f "%TBBURL%" "%DST%\%TBBPACKAGE%.zip" >nul
    if !errorlevel! neq 0 (
        echo tbb.bat : Error: Failed to download from %TBBURL%
        exit /B 1
    )

    rem Extract archive
    echo Extracting %TBBPACKAGE%.zip ...
    
    rem Try tar first (Windows 10+)
    tar -xf "%DST%\%TBBPACKAGE%.zip" -C "%DST%" 2>nul
    if !errorlevel! neq 0 (
        echo Tar extraction failed, attempting alternative method...
        
        rem Fallback: Use PowerShell for extraction
        powershell.exe -Command "Add-Type -Assembly 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('%DST%\%TBBPACKAGE%.zip', '%DST%')" 2>nul
        if !errorlevel! neq 0 (
            echo tbb.bat : Error: Failed to extract %DST%\%TBBPACKAGE%.zip
            echo Please ensure one of the following:
            echo   - Windows 10+ with tar.exe available
            echo   - PowerShell with System.IO.Compression support
            exit /B 1
        )
    )

    if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
        goto Build_oneTBB
    ) else (
        rem Copy extracted files for non-ARM64
        echo Copying extracted files ...
        xcopy "%DST%\oneapi-tbb-%TBBVERSION%\*" "%DST%\win\tbb" /E /I /Y >nul
        if !errorlevel! neq 0 (
            echo tbb.bat : Error: Failed to copy extracted files
            exit /B 1
        )
        if not exist "%DST%\win\tbb\redist\intel64\vc14" mkdir "%DST%\win\tbb\redist\intel64\vc14"
        goto Exit
    )

:Build_oneTBB
    echo Building oneTBB for ARM64 ...
    
    rem Detect and setup Visual Studio environment
    if "%VS_VER%"=="2026_build_tools" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
            @call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
        ) else (
            echo Error: Visual Studio 2026 Build Tools not found
            exit /B 1
        )
    ) else if "%VS_VER%"=="2019_build_tools" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
            @call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
        ) else (
            echo Error: Visual Studio 2019 Build Tools not found
            exit /B 1
        )
    ) else if "%VS_VER%"=="2017_build_tools" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
            @call "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" %PROCESSOR_ARCHITECTURE%
        ) else (
            echo Error: Visual Studio 2017 Build Tools not found
            exit /B 1
        )
    )

    pushd "%DST%\oneTBB-%TBBVERSION%"
    
    if exist "build-arm64" rmdir /s /q "build-arm64"
    
    echo Running CMake configure ...
    cmake -B build-arm64 -S . -GNinja ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DTBB_TEST=OFF ^
        -DCMAKE_INSTALL_PREFIX="%DST%\win\tbb"
    
    if !errorlevel! neq 0 (
        echo Error: CMake configuration failed
        popd
        exit /B 1
    )
    
    echo Building oneTBB ...
    cmake --build build-arm64
    
    if !errorlevel! neq 0 (
        echo Error: CMake build failed
        popd
        exit /B 1
    )
    
    echo Installing oneTBB ...
    cmake --install build-arm64
    
    if !errorlevel! neq 0 (
        echo Error: CMake install failed
        popd
        exit /B 1
    )
    
    rem Create directories for redist/lib
    if not exist "%DST%\win\tbb\redist\%PROCESSOR_ARCHITECTURE%\vc14" mkdir "%DST%\win\tbb\redist\%PROCESSOR_ARCHITECTURE%\vc14"
    if not exist "%DST%\win\tbb\lib\%PROCESSOR_ARCHITECTURE%\vc14" mkdir "%DST%\win\tbb\lib\%PROCESSOR_ARCHITECTURE%\vc14"
    
    rem Copy binaries and libraries
    echo Organizing output files ...
    robocopy "%DST%\win\tbb\bin" "%DST%\win\tbb\redist\%PROCESSOR_ARCHITECTURE%\vc14" /E
    robocopy "%DST%\win\tbb\bin" "%DST%\win\tbb\redist\vc14" /E
    robocopy "%DST%\win\tbb\bin" "%DST%\win\tbb\bin\vc14" /E
    robocopy "%DST%\win\tbb\lib" "%DST%\win\tbb\lib\%PROCESSOR_ARCHITECTURE%\vc14" *.lib
    robocopy "%DST%\win\tbb\lib" "%DST%\win\tbb\lib\vc14" *.lib
    
    popd
    exit /B 0

:Exit
    echo Downloaded and unpacked oneTBB small libraries to %DST%
    exit /B 0

) else (
    echo oneTBB small libraries are already installed in %DST%
    exit /B 0
)