# Whisper 字幕神器

這是一個在本機執行的 Whisper 字幕轉錄工具，提供可愛、直覺、手機也方便操作的前端介面，並用 Flask 提供上傳、轉錄、下載與環境檢查功能。現在可在 Windows 與 macOS 使用。

## 目前功能

- 上傳音訊或影片檔：`mp3`、`mp4`、`wav`、`m4a`、`ogg`、`webm`
- Whisper `medium` 模型轉錄
- 三種字幕切割模式：細緻、標準、寬鬆
- 下載 UTF-8 BOM 的 `.srt`
- 檢查 Python / Flask / openai-whisper / PyTorch / ffmpeg
- 協助安裝缺少的 pip 套件
- 查看 CPU / GPU 狀態，支援 Windows CUDA；macOS 可偵測 Apple MPS，但預設使用 CPU 以提高穩定性
- 轉錄時顯示百分比進度
- SRT 預覽以標準字幕區塊顯示，只預覽前幾段，完整內容請下載 `.srt`

## 啟動方式

### macOS

第一次使用建議先安裝 Python 3.11 / 3.12 與 ffmpeg：

```sh
brew install python@3.12 ffmpeg
```

一般使用請雙擊執行：

```text
Whisper 字幕神器.app
```

這個 `.app` 會在背景啟動本機 Flask 服務，不會顯示終端機視窗；啟動紀錄會寫到：

```text
logs/macos-app.log
```

#### 第一次使用 `.app` 會自動做什麼

- 尋找可用的 Python 3.9 以上版本
- 在專案內建立 `.venv`
- 啟用 pip
- 安裝 `requirements.txt` 裡的 Python 套件，例如 Flask 與 openai-whisper
- openai-whisper 會連帶安裝 PyTorch 等 Python 依賴，第一次下載可能會花比較久
- 啟動本機服務並開啟瀏覽器到 `http://localhost:5050`

#### `.app` 不會自動安裝什麼

- 不會安裝 macOS 系統層的 Python
- 不會安裝 Homebrew
- 不會自動安裝 `ffmpeg`

如果尚未安裝 `ffmpeg`，`.app` 會跳通知提醒；請先執行：

```sh
brew install ffmpeg
```

如果沒有可用的 Python，請先安裝：

```sh
brew install python@3.12
```

若要看完整啟動過程或除錯，也可以執行：

```sh
start.command
```

`start.command` 會自動建立 `.venv`、安裝 `requirements.txt`，並開啟：

- [http://localhost:5050](http://localhost:5050)

> macOS 的 `ControlCenter` / AirPlay Receiver 常會佔用 `5000`，所以 macOS 預設改用 `5050`。

若 macOS 擋下 `start.command`，可在終端機進入專案資料夾後執行：

```sh
chmod +x start.command
./start.command
```

若想指定其他 port：

```sh
WHISPER_PORT=5051 ./start.command
```

### Windows

直接執行：

```bat
start.bat
```

程式會優先找系統 Python；若找不到，會嘗試使用內建 Python。啟動後也會自動開啟 [http://localhost:5000](http://localhost:5000)。

## 安裝需求

- macOS 12 以上，或 Windows 10 / 11
- Python 3.11 或 3.12 與 pip
- `ffmpeg`
- 建議記憶體至少 8 GB
- macOS Apple Silicon 可偵測 PyTorch MPS，但 openai-whisper 在 MPS 上可能遇到不支援的 PyTorch operator，因此預設使用 CPU；Windows 若有 NVIDIA GPU，可搭配 CUDA 版 PyTorch 加速

> Windows 版 PyTorch 官方 wheel 主要支援 Python 3.9～3.12。若另一台電腦使用 Python 3.13 / 3.14，可能會出現找不到 `torchaudio` 或 CUDA 版 PyTorch 安裝檔的錯誤；建議改裝 Python 3.11 或 3.12。

> RTX 50 系列顯卡（例如 RTX 5060 Ti / `sm_120`）需要 CUDA 12.8 版 PyTorch wheel。若看到 `sm_120 is not compatible`，請在「裝置 / GPU」面板安裝推薦的 `cu128` 版本，或重新執行新版 `start.bat`。

> 若在 macOS 手動切到 Apple GPU (MPS) 後遇到 `SparseMPS` 或 `MPS backend` 錯誤，程式會自動卸載 MPS 模型並改用 CPU 重試同一個檔案。

## 手動安裝套件

```bat
python -m pip install -r requirements.txt
```

macOS 若使用專案虛擬環境：

```sh
.venv/bin/python -m pip install -r requirements.txt
```

## ffmpeg 提醒

macOS 建議：

```sh
brew install ffmpeg
```

Windows 如果系統 PATH 還沒有 `ffmpeg`，可以：

1. 使用 CapCut / 剪映內建的 ffmpeg
2. 或自行安裝到 `C:\ffmpeg\bin`
3. 並把路徑加到系統 PATH

## 目錄

- `app.py`：Flask 後端與 Whisper 工作流程
- `index.html`：前端操作頁
- `Whisper 字幕神器.app`：macOS 背景啟動器，不顯示終端機
- `start.command`：macOS 啟動腳本
- `start.bat`：Windows 啟動腳本
- `requirements.txt`：基本套件需求
- `uploads/`：暫存上傳檔案
- `memory.md`：這個專案的壓縮記憶
- `SKILL.md`：下次續接時的快速接手技能摘要

## 接手提醒

- 這個專案是 Whisper 字幕工具，不是抽獎系統
- 後續對話請一律使用中文
- 若要快速接手，先看 `memory.md`
