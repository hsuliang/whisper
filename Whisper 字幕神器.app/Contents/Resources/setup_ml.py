import urllib.request
import urllib.error
import zipfile
import tarfile
from pathlib import Path
import os
import shutil
import platform
import subprocess
import sys

def download_ffmpeg_static(update_callback):
    if shutil.which("ffmpeg"):
        update_callback("找到系統內建的 FFmpeg。")
        return True
    
    app_data = Path(os.environ.get("APP_DATA_DIR", Path.home() / "Library" / "Application Support" / "WhisperSubtitle"))
    app_data.mkdir(parents=True, exist_ok=True)
    bin_dir = app_data / "bin"
    bin_dir.mkdir(exist_ok=True)
    
    ffmpeg_path = bin_dir / "ffmpeg"
    if ffmpeg_path.exists():
        os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
        update_callback("找到已下載的 FFmpeg。")
        return True
    
    # Download based on architecture
    arch = platform.machine()
    if arch == "arm64":
        url = "https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/snapshot/ffmpeg.zip"
    else:
        url = "https://evermeet.cx/ffmpeg/get/zip"


    update_callback("正在為 Mac 下載 FFmpeg 靜態編譯檔...")
    try:
        zip_path = app_data / "ffmpeg.zip"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
        update_callback("FFmpeg 下載完成，正在解壓縮...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(bin_dir)
            
        zip_path.unlink()
        ffmpeg_bin = bin_dir / "ffmpeg"
        ffmpeg_bin.chmod(0o755)
        os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
        update_callback("FFmpeg 配置完成！")
        return True
    except Exception as e:
        update_callback(f"FFmpeg 下載失敗: {e}")
        return False

def install_ml_deps(update_callback):
    import importlib.util
    has_whisper = importlib.util.find_spec("whisper") is not None
    has_torch = importlib.util.find_spec("torch") is not None
    has_ytdlp = importlib.util.find_spec("yt_dlp") is not None
    
    if has_whisper and has_torch and has_ytdlp:
        update_callback("機器學習套件已準備就緒。")
        return True
        
    update_callback("正在下載機器學習套件 (Whisper, PyTorch)... 檔案較大，請耐心等候。")
    app_root = Path(os.environ.get("APP_ROOT_PATH", Path.cwd()))
    req_file = app_root / "requirements-ml.txt"
    if not req_file.exists():
        update_callback(f"找不到 {req_file}，略過套件安裝。")
        return False
        
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        line = line.strip()
        if line:
            update_callback(line)
            
    process.wait()
    if process.returncode == 0:
        update_callback("機器學習套件安裝成功！")
        return True
    else:
        update_callback("套件安裝失敗，請檢查網路連線或儲存空間。")
        return False
