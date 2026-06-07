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
rem
rem Windows analogue of merge_static_libs_lnx.tpl.sh: merges multiple static
rem archives (.lib / .obj inputs) into a single output archive using whichever
rem `lib`-compatible tool was discovered by the toolchain
rem (cc_toolchain_win.bzl populates `%{ar_path}` with `llvm-lib` if available
rem or with MSVC `lib` otherwise — same dispatch as the Linux template).
rem
rem Invoked from `_merge_static_libs` in dev/bazel/cc/link.bzl via the
rem `cpp_merge_static_libraries` action, exactly as the Linux side calls
rem `merge_static_libs.sh` through the same action.
rem
rem Usage: merge_static_libs.bat <output.lib> <input1> <input2> ...

setlocal EnableExtensions EnableDelayedExpansion

set "OUT=%~1"
if "%OUT%"=="" (
    echo merge_static_libs: missing output path
    exit /b 1
)

shift

rem `lib /OUT:<out> <inputs...>` joins archives and object files into one
rem archive. Both `llvm-lib` and MSVC `lib` accept the same flag spelling, so
rem the toolchain's choice is opaque to this script.
set "INPUTS="
:collect
if "%~1"=="" goto :run
set "INPUTS=!INPUTS! "%~1""
shift
goto :collect

:run
if "%INPUTS%"=="" (
    echo merge_static_libs: no input archives given for %OUT%
    exit /b 2
)

%{ar_path} /NOLOGO /OUT:"%OUT%" %INPUTS%
exit /b %ERRORLEVEL%
