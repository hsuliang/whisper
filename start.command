#!/bin/zsh
set -u

cd "$(dirname "$0")"

PORT=${WHISPER_PORT:-5050}
URL="http://localhost:${PORT}"
export WHISPER_PORT="${PORT}"

echo "================================"
echo " Whisper SRT - Starting on macOS"
echo "================================"
echo

export PATH="/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:${PATH}"

# 修正 Homebrew Python 3.12 的 pyexpat / libexpat 相容性問題 (Apple Silicon 與 Intel)
if command -v brew >/dev/null 2>&1; then
  EXPAT_PREFIX="$(brew --prefix expat 2>/dev/null)"
  if [ -z "$EXPAT_PREFIX" ] || [ ! -d "$EXPAT_PREFIX/lib" ]; then
    echo "[SETUP] Missing expat for Python. Installing via Homebrew..."
    brew install expat >/dev/null 2>&1 || true
    EXPAT_PREFIX="$(brew --prefix expat 2>/dev/null)"
  fi
  if [ -n "$EXPAT_PREFIX" ] && [ -d "$EXPAT_PREFIX/lib" ]; then
    export DYLD_LIBRARY_PATH="$EXPAT_PREFIX/lib:${DYLD_LIBRARY_PATH:-}"
  fi
fi

echo "[CHECK] Checking localhost:${PORT}..."
PIDS=$(lsof -tiTCP:${PORT} -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  for PID in ${(f)PIDS}; do
    CMD=$(ps -p "$PID" -o command= 2>/dev/null || true)
    if [[ "$CMD" == *"app.py"* ]]; then
      echo "[INFO] Stopping old Whisper server on port ${PORT}..."
      kill "$PID" 2>/dev/null || true
      sleep 1
    else
      echo "[ERROR] Port ${PORT} is used by another program:"
      echo "        ${CMD}"
      echo
      read "REPLY?Press Return to close this window..."
      exit 1
    fi
  done
fi
echo

BASE_PYTHON=""
if [ -x ".venv/bin/python" ]; then
  PYTHON="$PWD/.venv/bin/python"
  echo "[OK] Project virtual environment found"
else
  for CANDIDATE in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$CANDIDATE" >/dev/null 2>&1; then
      if "$CANDIDATE" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >/dev/null 2>&1; then
        BASE_PYTHON="$CANDIDATE"
        break
      fi
    fi
  done

  if [ -z "$BASE_PYTHON" ]; then
    echo "[ERROR] No compatible Python 3.9+ with venv support found."
    echo "        Install Python 3.11 or 3.12, then run start.command again."
    echo "        Homebrew example: brew install python@3.12"
    echo
    read "REPLY?Press Return to close this window..."
    exit 1
  fi

  echo "[SETUP] Creating project virtual environment with ${BASE_PYTHON}..."
  "$BASE_PYTHON" -m venv .venv 2>/dev/null || "$BASE_PYTHON" -m venv --without-pip .venv
  if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create .venv."
    echo
    read "REPLY?Press Return to close this window..."
    exit 1
  fi
  PYTHON="$PWD/.venv/bin/python"
fi

echo "[Python] ${PYTHON}"
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "[SETUP] Enabling pip..."
  "$PYTHON" -m ensurepip --upgrade 2>/dev/null || {
    echo "[SETUP] ensurepip failed, trying get-pip.py..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/_get_pip.py && "$PYTHON" /tmp/_get_pip.py 2>&1
  }
fi
echo

if ! "$PYTHON" -c "import flask, whisper" >/dev/null 2>&1; then
  echo "[INSTALL] Installing packages from requirements.txt..."
  echo "[INSTALL] First install may take a while because PyTorch is large."
  echo
  "$PYTHON" -m pip install -r requirements.txt
  if [ $? -ne 0 ]; then
    echo
    echo "[ERROR] Package install failed."
    echo "        Try: ${PYTHON} -m pip install -r requirements.txt"
    echo
    read "REPLY?Press Return to close this window..."
    exit 1
  fi
  echo
  echo "[DONE] Packages installed."
else
  echo "[OK] Flask and openai-whisper are installed."
fi
echo

if command -v ffmpeg >/dev/null 2>&1; then
  echo "[OK] ffmpeg found: $(command -v ffmpeg)"
else
  echo "[WARN] ffmpeg was not found in PATH."
  echo "       Install with Homebrew: brew install ffmpeg"
  echo "       The web app will also check common macOS paths."
fi
echo

echo "[START] Server starting at ${URL}"
echo "[INFO]  Browser opens in 3 seconds."
echo "[INFO]  Close this window to stop."
echo

(sleep 3 && open "${URL}") >/dev/null 2>&1 &
"$PYTHON" app.py
STATUS=$?

echo
if [ $STATUS -eq 0 ]; then
  echo "[STOP] Server closed."
else
  echo "[ERROR] Server stopped unexpectedly."
fi
echo
read "REPLY?Press Return to close this window..."
exit $STATUS
