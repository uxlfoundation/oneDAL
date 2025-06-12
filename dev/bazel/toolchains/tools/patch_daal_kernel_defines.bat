@echo off
REM patch_daal_kernel_defines.bat
REM %1 = input, %2 = output, %3 = cpu list

IF "%~3"=="" (
    copy "%~1" "%~2"
    exit /b
)

setlocal enabledelayedexpansion
set "input=%~1"
set "output=%~2"
set "cpus=%~3"

copy "%input%" "%output%.tmp"

for %%c in (%cpus%) do (
    powershell -Command "(Get-Content %output%.tmp) -replace '#define DAAL_KERNEL_%%c\b', '' | Set-Content %output%"
    move /Y %output% %output%.tmp > nul
)

move /Y %output%.tmp %output%
