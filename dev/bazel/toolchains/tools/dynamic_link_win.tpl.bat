@echo off
setlocal enabledelayedexpansion

rem --- Initialize variables ---
set "def_file="
set "args_patched="

rem --- Loop through all command-line arguments ---
:loop_args
if "%~1"=="" goto args_done

rem Check if argument ends with .def (case insensitive)
echo %~1 | findstr /i "\.def$" >nul
if not errorlevel 1 (
    rem Found .def file argument
    set "def_file=%~1"
) else (
    rem Add argument to args_patched
    if defined args_patched (
        set "args_patched=!args_patched! %~1"
    ) else (
        set "args_patched=%~1"
    )
)
shift
goto loop_args

:args_done

rem --- If a .def file was found ---
if defined def_file (
    rem Create patched def file name
    set "patched_def_file=%def_file%_patched"

    rem Overwrite/create patched def file with processed lines
    > "%patched_def_file%" (
        rem Read original def file line by line
        for /f "usebackq delims=" %%L in ("%def_file%") do (
            set "line=%%L"
            rem Skip lines starting with EXPORTS, or starting with ;, or empty lines
            echo !line! | findstr /i /r /c:"^EXPORTS" /c:"^;" >nul
            if errorlevel 1 (
                rem Line does not match, prepend "-u " and output it
                echo -u !line!
            )
        )
    )

    rem Run compiler with patched arguments
    %cc_path% %args_patched% @%patched_def_file%
) else (
    rem No .def file, run compiler with original arguments
    %cc_path% %*
)

endlocal
