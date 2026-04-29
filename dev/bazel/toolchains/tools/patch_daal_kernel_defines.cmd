@echo off
setlocal
%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0patch_daal_kernel_defines.ps1" %*
exit /b %ERRORLEVEL%
