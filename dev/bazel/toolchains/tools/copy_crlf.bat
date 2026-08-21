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
rem Copy a text file converting its line endings to CRLF.
rem
rem Usage: copy_crlf.bat <input> <output>
rem
rem Mirrors the makefile's Windows release staging, which pipes text files
rem through `sed -n -z -e 's/\r*\n/\r\n/g;p'`. Files are stored with LF in the
rem repository (see .gitattributes), so a plain copy would ship LF endings and
rem differ from the Make release on every line.

setlocal EnableExtensions

set "SRC=%~1"
set "DST=%~2"

if "%SRC%"=="" ( echo copy_crlf: missing input file & exit /b 1 )
if "%DST%"=="" ( echo copy_crlf: missing output file & exit /b 1 )

rem Collapse any existing CRLF to LF first so the conversion is idempotent,
rem exactly like the `\r*\n` in the sed expression above. Read and write raw
rem bytes to leave the rest of the content untouched.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$src = [System.IO.File]::ReadAllText('%SRC%');" ^
  "$lf = $src -replace \"`r`n\", \"`n\";" ^
  "$crlf = $lf -replace \"`n\", \"`r`n\";" ^
  "$enc = New-Object System.Text.UTF8Encoding($false);" ^
  "[System.IO.File]::WriteAllText('%DST%', $crlf, $enc)"
if errorlevel 1 (
    echo copy_crlf: failed to convert %SRC%
    exit /b 2
)

exit /b 0
