@echo off
setlocal enabledelayedexpansion

rem --- Check if at least one argument (output) is provided ---
if "%~1"=="" (
    echo Usage: %~nx0 output_file input_libs...
    exit /b 1
)

rem --- Get output file ---
set "output=%~1"
shift

rem --- Collect input libraries ---
set "input_libs="
:collect_inputs
if "%~1"=="" goto done_collecting
if defined input_libs (
    set "input_libs=!input_libs! %~1"
) else (
    set "input_libs=%~1"
)
shift
goto collect_inputs

:done_collecting

rem --- Build MRI script ---
setlocal disableDelayedExpansion
set "mri=CREATE %output%\n"
for %%L in (%input_libs%) do (
    setlocal enabledelayedexpansion
    set "mri=!mri!ADDLIB %%L\n"
    endlocal & set "mri=%mri%"
)
set "mri=%mri%SAVE\n"
endlocal & set "mri=%mri%"

rem --- Create temporary file for MRI commands ---
set "tmpfile=%TEMP%\mri_script.txt"
(
    echo CREATE %output%
    for %%L in (%input_libs%) do (
        echo ADDLIB %%L
    )
    echo SAVE
) > "%tmpfile%"

rem --- Run ar with MRI script ---
%lib_path% -M < "%tmpfile%"

rem --- Cleanup ---
del "%tmpfile%"

endlocal
