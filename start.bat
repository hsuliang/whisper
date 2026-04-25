@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Whisper

echo ================================
echo  Whisper SRT - Starting...
echo ================================
echo.

echo [CHECK] Checking localhost:5000...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$c=Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($c) { $p=Get-CimInstance Win32_Process -Filter ('ProcessId=' + $c.OwningProcess); if ($p.CommandLine -match 'app\.py') { Write-Host '[INFO] Stopping old Whisper server on port 5000...'; Stop-Process -Id $c.OwningProcess -Force; Start-Sleep -Seconds 1; exit 0 } else { Write-Host '[ERROR] Port 5000 is used by another program:'; Write-Host $p.CommandLine; exit 2 } }"
if errorlevel 2 (
    echo.
    echo [ERROR] Please close the program above, then run start.bat again.
    echo.
    pause
    exit /b 1
)
echo.

set PYTHON=

:: Prefer PyTorch-compatible Python versions on Windows.
for %%V in (3.12 3.11 3.10 3.9) do (
    py -%%V -m pip --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON=py -%%V
        echo [OK] Python %%V with pip found
        goto check_flask
    )
)

:: Test system python, but skip versions that PyTorch Windows wheels do not support.
python -m pip --version >nul 2>&1
if not errorlevel 1 (
    call :check_python_supported python
    if not errorlevel 1 (
        set PYTHON=python
        echo [OK] System Python with pip found
        goto check_flask
    )
    echo [SKIP] System Python found, but PyTorch on Windows supports Python 3.9 to 3.12.
    call :print_python_version python
)

:: Fallback: Claude Code built-in Python
set FALLBACK=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
if exist "%FALLBACK%" (
    "%FALLBACK%" -m pip --version >nul 2>&1
    if not errorlevel 1 (
        call :check_python_supported "%FALLBACK%"
        if not errorlevel 1 (
            set PYTHON="%FALLBACK%"
            echo [OK] Built-in Python found
            goto check_flask
        )
    )
)

echo.
echo [ERROR] No compatible Python + pip found!
echo.
echo Please install Python 3.11 or 3.12 from:
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
:: Check if CUDA is really usable. RTX 50 / Blackwell needs CUDA 12.8 PyTorch wheels.
call :cuda_usable %PYTHON%
if not errorlevel 1 (
    echo [GPU] CUDA detected - GPU acceleration enabled!
    goto start_server
)

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [GPU] No NVIDIA driver detected, using CPU mode.
    goto start_server
)

set CUDA_INDEX=https://download.pytorch.org/whl/cu121
set CUDA_LABEL=CUDA 12.1
%PYTHON% -c "import re, subprocess; out=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True, stderr=subprocess.STDOUT); raise SystemExit(0 if re.search(r'RTX\s*50|RTX\s*5\d\d\d|5060|5070|5080|5090', out, re.I) else 1)" >nul 2>&1
if not errorlevel 1 (
    set CUDA_INDEX=https://download.pytorch.org/whl/cu128
    set CUDA_LABEL=CUDA 12.8
    echo [GPU] RTX 50 series detected, using PyTorch CUDA 12.8 wheels.
)

:: CUDA not available or current PyTorch is incompatible - reinstall GPU PyTorch.
echo [GPU] Installing GPU version of PyTorch (%CUDA_LABEL%)...
echo [GPU] This may take a while (~2 GB), please wait...
echo.
%PYTHON% -m pip install torch --index-url %CUDA_INDEX% --upgrade --force-reinstall --progress-bar off
if errorlevel 1 (
    echo [WARN] GPU PyTorch install failed, using CPU mode.
) else (
    call :cuda_usable %PYTHON%
    if not errorlevel 1 (
        echo [GPU] GPU acceleration enabled!
    ) else (
        echo [WARN] PyTorch installed but CUDA test failed.
        echo [WARN] RTX 50 series requires PyTorch CUDA 12.8 and a recent NVIDIA driver.
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
exit /b

:check_python_supported
%* -c "import sys; v=sys.version_info; sys.exit(0 if v.major == 3 and 9 <= v.minor <= 12 else 1)" >nul 2>&1
exit /b %errorlevel%

:print_python_version
%* -c "import sys; print('[SKIP] Current version: Python ' + sys.version.split()[0])"
exit /b %errorlevel%

:cuda_usable
%* -c "import torch; ok=torch.cuda.is_available(); cc=torch.cuda.get_device_capability(0) if ok else tuple([0,0]); cv=tuple([int(x) for x in (torch.version.cuda or '0.0').split('.')[:2]]); ok=ok and not (cc >= tuple([12,0]) and cv < tuple([12,8])); torch.empty(1, device='cuda') if ok else None; torch.cuda.synchronize() if ok else None; sys_exit=__import__('sys').exit; sys_exit(0 if ok else 1)" >nul 2>&1
exit /b %errorlevel%
