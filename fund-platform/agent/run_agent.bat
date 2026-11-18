@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Run setup_windows.bat first.
    pause
    exit /b 1
)
if /i "%~1"=="safe" set "AGENT_FEISHU_DISABLED=1"
if /i "%~1"=="safe" echo Safe mode: real Feishu sends are disabled.
if "%~1"=="" goto :run
if /i "%~1"=="safe" goto :run
echo Unknown mode: %~1
echo Usage: run_agent.bat [safe]
pause
exit /b 2
:run
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1
set "AGENT_EXIT_CODE=%ERRORLEVEL%"
if not "%AGENT_EXIT_CODE%"=="0" pause
exit /b %AGENT_EXIT_CODE%
