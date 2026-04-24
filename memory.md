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

## 目前檔案重點

- `app.py`
  - Flask 主程式
  - Whisper lazy loading
  - ffmpeg 自動掃描
  - GPU / CPU 裝置切換
  - 背景執行轉錄與安裝工作
- `index.html`
  - 單頁前端
  - 上傳、進度、結果預覽、下載
  - 環境檢查 / 安裝協助 / 裝置資訊 Modal
- `start.bat`
  - Windows 啟動入口
  - 會先找 Python，再啟動 `app.py`
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

## 重要偏好

- 之後一律使用中文回覆使用者
- 專案主題是 Whisper，不要再切回抽獎系統

## 下次若要續做

- 先跑 `start.bat` 做實機測試
- 優先檢查：
  - 上傳是否成功
  - 轉錄輪詢是否正常
  - SRT 下載是否正常
  - 裝置切換與 CUDA 安裝面板是否正常
- 若使用者回報問題，先看 `app.py` API 與 `index.html` fetch 對應是否一致

