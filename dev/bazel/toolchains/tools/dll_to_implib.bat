@echo off
rem ============================================================================
rem Copyright contributors to the oneDAL project
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem ============================================================================
rem
rem Generate a DLL's import library (.lib) from the DLL itself by harvesting
rem its exports via dumpbin and feeding them to lib /def:.
rem
rem Usage: dll_to_implib.bat <input.dll> <output.lib> <output.exp>
rem
rem Bazel cannot intercept the import lib that lld-link writes as a side
rem effect of /DLL link, so release.bzl invokes this script as a follow-up
rem action that takes the freshly-built DLL as a Bazel-tracked input and
rem produces the import lib as a Bazel-tracked output.

setlocal EnableExtensions EnableDelayedExpansion

set "DLL_IN=%~1"
set "LIB_OUT=%~2"
set "EXP_OUT=%~3"

if "%DLL_IN%"=="" ( echo dll_to_implib: missing input dll & exit /b 1 )
if "%LIB_OUT%"=="" ( echo dll_to_implib: missing output lib & exit /b 1 )

rem Bazel actions get a stripped PATH that does not include MSVC tooling.
rem Discover dumpbin/lib via VCTOOLSINSTALLDIR (set by VsDevCmd). Fall
rem back to vswhere -find to locate the latest installed MSVC bin dir.
if defined VCTOOLSINSTALLDIR (
    set "PATH=%VCTOOLSINSTALLDIR%bin\HostX64\x64;%PATH%"
)
where dumpbin >NUL 2>&1
if errorlevel 1 (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq delims=" %%v in (`"!VSWHERE!" -latest -products * ^
              -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
              -find "VC\Tools\MSVC\*\bin\HostX64\x64\dumpbin.exe"`) do (
            set "PATH=%%~dpv;!PATH!"
        )
    )
)
where dumpbin >NUL 2>&1
if errorlevel 1 (
    echo dll_to_implib: cannot locate dumpbin.exe ^(set VCTOOLSINSTALLDIR or install VS Build Tools^)
    exit /b 2
)

set "DEF_TMP=%LIB_OUT%.def"
set "DLL_NAME=%~nx1"

> "%DEF_TMP%" echo LIBRARY %DLL_NAME%
>>"%DEF_TMP%" echo EXPORTS

rem dumpbin /exports prints lines like:
rem   <ord>   <hint>   <rva>   <name>
rem Keep just <name> (the 4th token) for entries whose first token is a
rem decimal ordinal. findstr drops headers/footers; the for /f loop
rem iterates dumpbin's output via a sub-shell so we do not depend on a
rem temp file.
for /f "tokens=4" %%n in ('dumpbin /nologo /exports "%DLL_IN%" ^| findstr /R /C:"^ *[0-9][0-9]* "') do (
    >>"%DEF_TMP%" echo     %%n
)

if not exist "%DEF_TMP%" (
    echo dll_to_implib: failed to write %DEF_TMP%
    exit /b 3
)

set "LIB_TOOL=lib"
where llvm-lib >NUL 2>&1
if not errorlevel 1 set "LIB_TOOL=llvm-lib"

%LIB_TOOL% /nologo /machine:x64 /def:"%DEF_TMP%" /out:"%LIB_OUT%" /name:%DLL_NAME%
if errorlevel 1 (
    echo dll_to_implib: %LIB_TOOL% /def failed
    exit /b 4
)

rem `lib /def:` writes both <out>.lib and <out>.exp; rename .exp to where
rem Bazel expects it (Bazel was given LIB_OUT and EXP_OUT separately).
set "LIB_EXP=%LIB_OUT:.lib=.exp%"
if not "%EXP_OUT%"=="" if /I not "%LIB_EXP%"=="%EXP_OUT%" (
    if exist "%LIB_EXP%" move /Y "%LIB_EXP%" "%EXP_OUT%" >NUL
)

del /Q "%DEF_TMP%" 2>NUL
exit /b 0
