@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Whisper

echo ================================
echo  Whisper SRT - Starting...
echo ================================
echo.

set PYTHON=

:: Test system python (skip Windows Store stub)
python -m pip --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON=python
    echo [OK] System Python with pip found
    goto check_flask
)

:: Fallback: Claude Code built-in Python
set FALLBACK=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
if exist "%FALLBACK%" (
    "%FALLBACK%" -m pip --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON="%FALLBACK%"
        echo [OK] Built-in Python found
        goto check_flask
    )
)

echo.
echo [ERROR] No working Python + pip found!
echo.
echo Please install Python from:
echo   https://www.python.org/downloads/
echo.
echo IMPORTANT: Check "Add Python to PATH" during install.
echo After installing, run this file again.
echo.
pause
exit /b 1

:check_flask
echo [Python] %PYTHON%
echo.
%PYTHON% -c "import flask" >nul 2>&1
if not errorlevel 1 goto check_gpu

echo [INSTALL] Installing packages (first time, ~2-3 GB download)...
echo [INSTALL] Do NOT close this window...
echo.
%PYTHON% -m pip install flask openai-whisper
if errorlevel 1 (
    echo.
    echo [ERROR] Install failed.
    echo Try: %PYTHON% -m pip install flask openai-whisper
    echo.
    pause
    exit /b 1
)
echo.
echo [DONE] Packages installed!
echo.

:check_gpu
:: Check if CUDA (GPU) is available
%PYTHON% -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [GPU] CUDA detected - GPU acceleration enabled!
    goto start_server
)

:: CUDA not available - reinstall PyTorch with CUDA 12.1 support
echo [GPU] Installing GPU version of PyTorch (CUDA 12.1)...
echo [GPU] This may take a while (~2 GB), please wait...
echo.
%PYTHON% -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade
if errorlevel 1 (
    echo [WARN] GPU PyTorch install failed, using CPU mode.
) else (
    %PYTHON% -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if not errorlevel 1 (
        echo [GPU] GPU acceleration enabled!
    ) else (
        echo [WARN] PyTorch installed but CUDA not detected.
        echo [WARN] Possible reasons: no NVIDIA GPU, or driver too old.
        echo [WARN] Continuing in CPU mode...
    )
)
echo.

:start_server
echo [START] Server starting at http://localhost:5000
echo [INFO]  Browser opens in 3 seconds.
echo [INFO]  Close this window to stop.
echo.

start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

%PYTHON% app.py

echo.
if errorlevel 1 (
    echo [ERROR] Server stopped unexpectedly.
) else (
    echo [STOP] Server closed.
)
echo.
pause
