@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dpc_link_win.ps1" "%{dpcc}" %*
exit /b %ERRORLEVEL%
