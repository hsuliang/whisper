# 專案壓縮記憶

## 專案定位

- 專案名稱：Whisper 字幕神器
- 類型：本機執行的 Whisper 字幕轉錄工具
- 架構：`Flask + 單頁 index.html`
- 主要用途：上傳音訊或影片，交給 Whisper 轉錄，下載 `.srt`

## 目前可用功能

- 支援上傳格式：`mp3`、`mp4`、`wav`、`m4a`、`ogg`、`webm`
- 前端可拖拉上傳或點擊選檔
- 三種字幕切割模式：`fine`、`standard`、`coarse`
- 後端提供：
  - `/upload`
  - `/status/<job_id>`
  - `/download/<job_id>`
  - `/cancel/<job_id>`
  - `/env-check`
  - `/install`
  - `/install-cuda-torch`
  - `/device-info`
  - `/set-device`
  - `/unload-model`
  - `/cuda-diagnose`
- 下載字幕格式為 `UTF-8 with BOM`，方便 Windows 字幕軟體開啟
- 已加入 macOS 啟動入口 `start.command`，會自動建立 `.venv`、安裝需求並開啟瀏覽器
- macOS 預設使用 `http://localhost:5050`，避開 `ControlCenter` / AirPlay Receiver 常佔用的 `5000`
- macOS Apple Silicon 可偵測 PyTorch MPS，但因 openai-whisper 可能遇到 `SparseMPS` / MPS backend 不支援，預設使用 CPU；Windows/NVIDIA 仍維持 CUDA 偵測與安裝協助
- Windows 建議使用 Python 3.11 或 3.12；避免 Python 3.13 / 3.14 找不到 PyTorch CUDA wheel
- RTX 50 系列 / Blackwell (`sm_120`) 需要 PyTorch CUDA 12.8 (`cu128`)，不能使用舊的 `cu121`

## 目前檔案重點

- `app.py`
  - Flask 主程式
  - Whisper lazy loading
  - ffmpeg 自動掃描，支援 Windows 常見路徑與 macOS Homebrew 常見路徑
  - GPU / CPU 裝置切換，支援 CUDA 與 Apple MPS
  - 背景執行轉錄與安裝工作
- `index.html`
  - 單頁前端
  - 上傳、進度、結果預覽、下載
  - 環境檢查 / 安裝協助 / 裝置資訊 Modal
- `start.bat`
  - Windows 啟動入口
  - 會優先找 Python 3.12 / 3.11 / 3.10 / 3.9，再啟動 `app.py`
- `start.command`
  - macOS 啟動入口
  - 會建立 `.venv`，使用 Python 3.12 / 3.11 / 3.10 / 3.9 / python3，安裝 `requirements.txt` 後啟動 `app.py`
- `README.md`
  - 已改成目前 Whisper 專案的說明

## 這次已完成事項

- 已確認本專案應忽略抽獎系統需求，專注 Whisper
- 已重建 `index.html`
- 已重整 `app.py`
- 已更新 `README.md`
- 已用內建 Python 驗證：
  - `py_compile app.py` 通過
  - `import app` 通過
- 啟動偵測結果：
  - 可找到 CapCut 內建 `ffmpeg`
  - 有偵測到 `NVIDIA GeForce RTX 3060`
- 針對另一台電腦 Python 3.14 安裝 CUDA PyTorch 失敗：
  - `/install-cuda-torch` 改成只安裝 Whisper 需要的 `torch`
  - 偵測到不相容 Python 時，前端會提示改用 Python 3.11 / 3.12
  - `start.bat` 不再直接拿 Python 3.14 執行安裝流程
- 針對 RTX 5060 Ti 顯示 `sm_120 is not compatible`：
  - CUDA 偵測改為實際建立 CUDA tensor，不只看 `torch.cuda.is_available()`
  - `start.bat` 偵測 RTX 50 系列時改裝 `https://download.pytorch.org/whl/cu128`
  - `/cuda-diagnose` 會回傳顯卡架構、PyTorch CUDA runtime、推薦索引與相容性問題
- 針對瀏覽器仍打到舊版 Flask：
  - 已停止舊的 `python app.py` 佔用程序
  - `start.bat` 啟動時會檢查 localhost:5000，若是舊 `app.py` 會先停止
  - `/` 回應加入 `Cache-Control: no-store`，避免前端 HTML 快取舊邏輯
- 針對 macOS 版本：
  - 新增 `start.command`，可在 macOS 雙擊或終端機執行
  - `start.command` 預設使用 `WHISPER_PORT=5050`；可用 `WHISPER_PORT=5051 ./start.command` 指定其他 port
  - `app.py` 啟動 port 改讀 `WHISPER_PORT` / `PORT`，不再硬寫 `5000`
  - `app.py` 新增平台偵測、macOS ffmpeg 常見路徑與 Apple MPS 偵測
  - 因 Whisper 在 MPS 上可能遇到 PyTorch sparse operator 不支援，macOS 預設不再自動選 MPS；若使用者手動切 MPS 失敗，會自動改用 CPU 重試
  - `/set-device` 現在支援 `cpu`、`cuda`、`mps`
  - 前端「裝置 / GPU」面板會依平台顯示 CUDA 或 MPS 狀態，macOS 不再提示安裝 CUDA 版 PyTorch
  - `README.md`、`AGENTS.md`、`SKILL.md` 已更新為 Windows / macOS 雙平台

## 重要偏好

- 之後一律使用中文回覆使用者
- 專案主題是 Whisper，不要再切回抽獎系統

## 下次若要續做

- macOS 先跑 `start.command` 做實機測試；Windows 先跑 `start.bat`
- 優先檢查：
  - 上傳是否成功
  - 轉錄輪詢是否正常
  - SRT 下載是否正常
  - 裝置切換與 GPU / CUDA / MPS 面板是否正常
- 若使用者回報問題，先看 `app.py` API 與 `index.html` fetch 對應是否一致
