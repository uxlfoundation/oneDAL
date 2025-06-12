@echo off
setlocal enabledelayedexpansion

set "input=%~1"
set "output=%~2"
set "cpus=%*"
set "cpus=%cpus:*%~2 =%"

copy /Y "%input%" "%output%.tmp"

for %%c in (%cpus%) do (
    > "%output%" (
        for /f "usebackq delims=" %%l in ("%output%.tmp") do (
            echo %%l | findstr /r /c:"#define DAAL_KERNEL_%%c" >nul
            if errorlevel 1 echo %%l
        )
    )
    move /Y "%output%" "%output%.tmp" >nul
)

move /Y "%output%.tmp" "%output%"
endlocal
