#!/bin/zsh
set -u

# Get the path to the Resources directory (where this script is located)
APP_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_ROOT"

# Define Data Directory
DATA_DIR="$HOME/Library/Application Support/WhisperSubtitle"
LOG_DIR="$DATA_DIR/logs"
VENV_DIR="$DATA_DIR/venv"

mkdir -p "$LOG_DIR" || exit 1
LOG_FILE="$LOG_DIR/macos-app.log"
exec >> "$LOG_FILE" 2>&1

echo "================================"
echo "Whisper 字幕神器 launch $(date)"
echo "App Root: $APP_ROOT"
echo "Data Dir: $DATA_DIR"
echo "================================"

export PATH="/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:${PATH}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export APP_ROOT_PATH="$APP_ROOT"
export APP_DATA_DIR="$DATA_DIR"

# --- 0. Check App Location ---
BUNDLE_DIR="$(cd "$APP_ROOT/../.." && pwd)"
if [[ "$BUNDLE_DIR" != "/Applications"* ]] && [[ "$BUNDLE_DIR" != "$HOME/Applications"* ]]; then
  MOVE_PROMPT=$(/usr/bin/osascript -e '
    set dialogText to "為了獲得最佳效能與穩定性，強烈建議將「Whisper 字幕神器」移動到「應用程式」資料夾中執行。\n\n是否讓程式現在自動幫您移動？"
    try
      set response to display dialog dialogText buttons {"稍後再說", "自動移動並重啟"} default button "自動移動並重啟" with title "Whisper 字幕神器" with icon note
      return button returned of response
    on error
      return "cancel"
    end try
  ')
  
  if [ "$MOVE_PROMPT" = "自動移動並重啟" ]; then
    echo "[SETUP] User chose to move app to /Applications"
    MOVE_RESULT=$(/usr/bin/osascript -e "
      try
        tell application \"Finder\"
          set appAlias to POSIX file \"$BUNDLE_DIR\" as alias
          set destFolder to POSIX file \"/Applications\" as alias
          move appAlias to destFolder with replacing
        end tell
        return \"success\"
      on error errMsg
        return \"error: \" & errMsg
      end try
    ")
    
    if [[ "$MOVE_RESULT" == *"success"* ]]; then
      APP_NAME=$(basename "$BUNDLE_DIR")
      DEST_APP="/Applications/$APP_NAME"
      echo "[SETUP] Moved successfully, relaunching $DEST_APP"
      open "$DEST_APP"
      exit 0
    else
      echo "[ERROR] Move failed: $MOVE_RESULT"
      /usr/bin/osascript -e 'display dialog "自動移動失敗，請您手動將程式拖曳至「應用程式」資料夾中。" buttons {"確定"} default button "確定" with title "移動失敗" with icon caution' >/dev/null 2>&1
    fi
  fi
fi


# --- 1. Python Pre-flight Check ---
BASE_PYTHON=""
for CANDIDATE in python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v "$CANDIDATE" >/dev/null 2>&1; then
    if "$CANDIDATE" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >/dev/null 2>&1; then
      BASE_PYTHON="$CANDIDATE"
      break
    fi
  fi
done

if [ -z "$BASE_PYTHON" ]; then
  echo "[ERROR] Python 3.9+ not found."
  /usr/bin/osascript -e '
    set dialogText to "【缺少 Python 核心套件】\n\n要啟動 Whisper 字幕神器，您的 Mac 必須安裝 Python 3.9 或以上版本。\n\n👉 解決方式：\n請點擊下方按鈕前往官網下載並安裝 Python，完成後再重新啟動本程式。"
    set response to display dialog dialogText buttons {"離開", "前往下載 Python"} default button "前往下載 Python" with title "Whisper 字幕神器" with icon stop
    if button returned of response is "前往下載 Python" then
      open location "https://www.python.org/downloads/macos/"
    end if
  ' >/dev/null 2>&1
  exit 1
fi

# --- 2. Environment Setup ---
if [ -x "$VENV_DIR/bin/python" ]; then
  PYTHON="$VENV_DIR/bin/python"
  echo "[OK] Virtual environment found at $VENV_DIR"
else
  echo "[SETUP] Creating venv at $VENV_DIR with ${BASE_PYTHON}..."
  "$BASE_PYTHON" -m venv --clear "$VENV_DIR" 2>/dev/null || "$BASE_PYTHON" -m venv --without-pip --clear "$VENV_DIR" || {
    echo "[ERROR] Failed to create venv."
    /usr/bin/osascript -e 'display dialog "無法建立 Python 虛擬環境！\n\n請確認您的 Mac 有足夠的磁碟空間與權限。" buttons {"了解"} default button "了解" with title "錯誤" with icon stop' >/dev/null 2>&1
    exit 1
  }
  PYTHON="$VENV_DIR/bin/python"
fi

if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "[SETUP] Enabling pip..."
  "$PYTHON" -m ensurepip --upgrade 2>/dev/null || {
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/_get_pip.py && "$PYTHON" /tmp/_get_pip.py 2>&1 || {
      echo "[ERROR] Failed to enable pip."
      exit 1
    }
  }
fi

# Install ONLY core requirements (pywebview, flask) silently and quickly
if ! "$PYTHON" -c "import flask, webview" >/dev/null 2>&1; then
  echo "[INSTALL] Installing requirements-core.txt..."
  "$PYTHON" -m pip install -r "$APP_ROOT/requirements-core.txt" || {
    echo "[ERROR] Core package install failed."
    /usr/bin/osascript -e 'display dialog "無法安裝核心啟動套件，請檢查網路連線。" buttons {"OK"} default button "OK" with title "Whisper 字幕神器" with icon stop' >/dev/null 2>&1
    exit 1
  }
fi

# --- 3. Start App ---
echo "[START] Executing app.py..."
# We pass the port selection logic to python, or we can just pick one here.
# But with pywebview, we don't necessarily need an exposed port if we use an ephemeral one or static.
# We'll just let python handle it.
cd "$APP_ROOT"
exec "$PYTHON" app.py
