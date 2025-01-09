@echo off
rem ============================================================================
rem Copyright 2014 Intel Corporation
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
call:GetFullPath "%~dp0.."       DAAL
call:GetFullPath "%~dp0"         SCRIPT_PATH
call:GetFullPath "%~dp0..\..\.." DAAL_UP
call:GetFullPath "%~dp0..\.."    DAAL_UP_OLD

set DAAL_IA=intel64

:ParseArgs
if /i "%1"=="" goto :CheckLayout
if /i "%1"=="intel64" (set DAAL_IA=intel64) & shift & goto :ParseArgs
shift
goto :ParseArgs

:CheckLayout
if "%SCRIPT_PATH%"=="%DAAL%\env\" (
  goto :GoodArgs
) else (
  set "DALROOT=%ONEAPI_ROOT%"
  set "INCLUDE=%ONEAPI_ROOT%\include\dal;%INCLUDE%"
  set "CPATH=%ONEAPI_ROOT%\include;%ONEAPI_ROOT%\include\dal;%CPATH%"
  goto :GoodArgs2024
)

:GetFullPath
set %2=%~f1
exit /b 0

:Usage
echo.
echo Syntax:  call %~nx0 [^<arch^>]
echo Where ^<arch^> is one of
echo   intel64  - setup environment for Intel(R) 64 architecture
echo default is intel64
exit /b 0

:GoodArgs
set "DALROOT=%DAAL%"
set "CPATH=%DAAL%\include;%DAAL%\include\dal;%CPATH%"
if exist "%DAAL%\include\dal" (
  set "INCLUDE=%DAAL%\include\dal;%INCLUDE%"
  set "LIB=%DAAL%\lib;%LIB%"
  set "CLASSPATH=%DAAL%\share\java\onedal.jar;%CLASSPATH%"
  set "PATH=%DAAL%\bin;%PATH%"
) else (
  set "INCLUDE=%DAAL%\include;%INCLUDE%"
  set "LIB=%DAAL%\lib\%DAAL_IA%;%LIB%"
  set "CLASSPATH=%DAAL%\lib\onedal.jar;%CLASSPATH%"
  if exist "%DAAL_UP_OLD%\redist" (
      set "PATH=%DAAL_UP_OLD%\redist\%DAAL_IA%_win\daal;%PATH%"
  ) else (
      set "PATH=%DAAL%\redist\%DAAL_IA%;%PATH%"
  )
)
set "CMAKE_PREFIX_PATH=%DAAL%;%CMAKE_PREFIX_PATH%"
set "PKG_CONFIG_PATH=%DAAL%\lib\pkgconfig;%PKG_CONFIG_PATH%"

endlocal& ^
set DAL_MAJOR_BINARY=__DAL_MAJOR_BINARY__& ^
set DAL_MINOR_BINARY=__DAL_MINOR_BINARY__& ^
set DALROOT=%DALROOT%& ^
set INCLUDE=%INCLUDE%& ^
set CPATH=%CPATH%& ^
set LIB=%LIB%& ^
set PATH=%PATH%& ^
set LD_LIBRARY_PATH=%LD_LIBRARY_PATH%& ^
set CLASSPATH=%CLASSPATH%& ^
set CMAKE_PREFIX_PATH=%CMAKE_PREFIX_PATH%& ^
set PKG_CONFIG_PATH=%PKG_CONFIG_PATH%& ^
goto:eof

:GoodArgs2024
endlocal& ^
set DALROOT=%DALROOT%& ^
set CLASSPATH=%CLASSPATH%& ^
set INCLUDE=%INCLUDE%& ^
set CPATH=%CPATH%& ^
goto:eof
