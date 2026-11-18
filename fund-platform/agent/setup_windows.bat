@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher "py" not found. Install Python 3.12 x64 first.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" goto :have_venv
echo Creating .venv with Python 3.12...
py -3.12 -m venv .venv
if errorlevel 1 goto :fail
:have_venv

set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

rem uvloop has no Windows wheels; uvicorn uses the asyncio loop instead.
findstr /V /R "^uvloop" requirements.txt > ".requirements-windows.tmp"
"%PYTHON%" -m pip install -r ".requirements-windows.tmp"
if errorlevel 1 goto :fail_cleanup
del ".requirements-windows.tmp"

if /i "%~1"=="rag" goto :install_rag
goto :done

:install_rag
echo Installing RAG extras with CPU-only torch...
"%PYTHON%" -m pip install -r requirements-rag.txt --extra-index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail

:done
if not exist ".env" (
    echo.
    echo Next: copy .env.example to .env and fill keys. Do not commit .env.
)
echo.
echo Install complete. Safe start: run_agent.bat safe
exit /b 0

:fail_cleanup
del ".requirements-windows.tmp"
:fail
echo.
echo Install failed. See messages above.
pause
exit /b 1
