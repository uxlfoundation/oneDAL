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
 
setlocal enabledelayedexpansion
 
set TBBVERSION=2023.0.0
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
    set "TBBURL=https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v%TBBVERSION%.zip"
    set "TBBPACKAGE="
) else (
    set "TBBURL=https://github.com/uxlfoundation/oneTBB/releases/download/v%TBBVERSION%/oneapi-tbb-%TBBVERSION%-win.zip"
    set "TBBPACKAGE=oneapi-tbb-%TBBVERSION%-win"
)
 
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
    curl -L -o "%DST%\%TBBPACKAGE%.zip" "%TBBURL%"
    if !errorlevel! neq 0 (
        echo tbb.bat : Error: Failed to download from %TBBURL%
        echo Make sure curl is installed and accessible
        exit /B 1
    )
 
    rem Extract archive
    echo Extracting %TBBPACKAGE%.zip ...
    pushd "%DST%"
    tar -xf "%TBBPACKAGE%.zip" || (
        echo tbb.bat : Error: Failed to extract archive
        echo Make sure tar.exe is available ^(Windows 10+ or git bash^)
        popd
        exit /B 1
    )
    popd
 
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
    
    rem Auto-detect Visual Studio installation
    set "VS_FOUND=0"
    set "VCVARS_PATH="
    
    rem Check Visual Studio 2022 (version 17)
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        set "VS_FOUND=1"
        echo Found Visual Studio 2022 Build Tools
    )
    
    rem Check Visual Studio 2019 (version 16)
    if "!VS_FOUND!"=="0" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
            set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
            set "VS_FOUND=1"
            echo Found Visual Studio 2019 Build Tools
        )
    )
    
    rem Check Visual Studio 2017 (version 15)
    if "!VS_FOUND!"=="0" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
            set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
            set "VS_FOUND=1"
            echo Found Visual Studio 2017 Build Tools
        )
    )
    
    rem Check with Enterprise/Professional/Community editions
    if "!VS_FOUND!"=="0" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
            set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
            set "VS_FOUND=1"
            echo Found Visual Studio 2022 Enterprise
        )
    )
    
    if "!VS_FOUND!"=="0" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
            set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
            set "VS_FOUND=1"
            echo Found Visual Studio 2019 Enterprise
        )
    )
    
    if "!VS_FOUND!"=="0" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
            set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
            set "VS_FOUND=1"
            echo Found Visual Studio 2022 Professional
        )
    )
    
    if "!VS_FOUND!"=="0" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
            set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat"
            set "VS_FOUND=1"
            echo Found Visual Studio 2019 Professional
        )
    )
    
    if "!VS_FOUND!"=="0" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
            set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
            set "VS_FOUND=1"
            echo Found Visual Studio 2022 Community
        )
    )
    
    if "!VS_FOUND!"=="0" (
        if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" (
            set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
            set "VS_FOUND=1"
            echo Found Visual Studio 2019 Community
        )
    )
    
    rem If still not found, check environment variable override
    if "!VS_FOUND!"=="0" (
        if not "!VSINSTALLDIR!"=="" (
            set "VCVARS_PATH=!VSINSTALLDIR!\VC\Auxiliary\Build\vcvarsall.bat"
            if exist "!VCVARS_PATH!" (
                set "VS_FOUND=1"
                echo Found Visual Studio via VSINSTALLDIR environment variable
            )
        )
    )
    
    rem If still not found, error out
    if "!VS_FOUND!"=="0" (
        echo Error: No Visual Studio Build Tools or IDE installation found
        echo Supported versions: 2017, 2019, 2022
        echo Please install Visual Studio Build Tools or set VSINSTALLDIR environment variable
        exit /B 1
    )
    
    rem Setup Visual Studio environment
    @call "!VCVARS_PATH!" %PROCESSOR_ARCHITECTURE%
    if !errorlevel! neq 0 (
        echo Error: Failed to setup Visual Studio environment
        exit /B 1
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