from __future__ import annotations

import gc
import importlib
import os
import platform
import shutil
import subprocess
import sys
import threading
import uuid
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file


# 中文註解：這個專案以單機使用為主，後端只負責接檔、呼叫 Whisper、回傳結果。
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_FILE = BASE_DIR / "index.html"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp3", "mp4", "wav", "m4a", "ogg", "webm"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024
PYTORCH_WINDOWS_PYTHON_MIN = (3, 9)
PYTORCH_WINDOWS_PYTHON_MAX = (3, 12)
PYTORCH_PIP_PACKAGES = {"openai-whisper", "whisper", "torch", "torchvision", "torchaudio"}
CUDA_INDEX_CU121 = "https://download.pytorch.org/whl/cu121"
CUDA_INDEX_CU128 = "https://download.pytorch.org/whl/cu128"
MPS_ERROR_MARKERS = (
    "SparseMPS",
    "MPS backend",
    "AutogradMPS",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

app = Flask(__name__, static_folder=".", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@dataclass
class JobState:
    status: str
    filename: str
    file_path: str | None = None
    srt: str | None = None
    error_msg: str | None = None
    progress: int = 0
    progress_text: str = "等待處理"


@dataclass
class InstallJobState:
    status: str = "running"
    lines: list[str] = field(default_factory=list)


jobs: dict[str, JobState] = {}
jobs_lock = threading.Lock()

install_jobs: dict[str, InstallJobState] = {}
install_lock = threading.Lock()

_whisper_model = None
_model_lock = threading.Lock()
_transcribe_lock = threading.Lock()
DEVICE = "cpu"
USE_FP16 = False


def is_windows() -> bool:
    return os.name == "nt"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def platform_label() -> str:
    if is_windows():
        return "Windows"
    if is_macos():
        return "macOS"
    return platform.system() or "Unknown"


def restart_entry_name() -> str:
    if is_windows():
        return "start.bat"
    if is_macos():
        return "start.command"
    return "python app.py"


def server_port() -> int:
    raw_port = os.environ.get("WHISPER_PORT") or os.environ.get("PORT") or "5000"
    try:
        port = int(raw_port)
    except ValueError:
        print(f"[WARN] 無效的 port：{raw_port}，改用 5000。")
        return 5000

    if not 1 <= port <= 65535:
        print(f"[WARN] port 超出範圍：{port}，改用 5000。")
        return 5000

    return port


def mps_requested() -> bool:
    return os.environ.get("WHISPER_ENABLE_MPS", "").strip().lower() in {"1", "true", "yes", "on"}


def python_version_text() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def is_pytorch_python_supported() -> bool:
    major_minor = sys.version_info[:2]
    if is_windows():
        return PYTORCH_WINDOWS_PYTHON_MIN <= major_minor <= PYTORCH_WINDOWS_PYTHON_MAX
    return major_minor >= PYTORCH_WINDOWS_PYTHON_MIN


def python_support_blocker() -> str | None:
    if is_pytorch_python_supported():
        return None

    if is_windows():
        return (
            f"目前使用 Python {python_version_text()}。Windows 版 PyTorch 官方 wheel 支援 "
            f"Python 3.9～3.12；建議安裝 Python 3.11 或 3.12 後重新執行 {restart_entry_name()}。"
        )

    return f"目前使用 Python {python_version_text()}，請改用 Python 3.9 以上。"


def package_base_name(package: str) -> str:
    name = str(package).strip().lower().replace("_", "-")
    name = name.split("[", 1)[0]
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        name = name.split(separator, 1)[0]
    return name.strip()


def packages_need_pytorch(packages: list[str]) -> bool:
    return any(package_base_name(package) in PYTORCH_PIP_PACKAGES for package in packages)


def version_tuple(version: str | None) -> tuple[int, int]:
    parts: list[int] = []
    for part in str(version or "").split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
        if len(parts) == 2:
            break
    while len(parts) < 2:
        parts.append(0)
    return parts[0], parts[1]


def capability_text(capability: tuple[int, int] | None) -> str:
    if not capability:
        return "未知"
    return f"sm_{capability[0]}{capability[1]}"


def looks_like_rtx50(gpu_name: str | None) -> bool:
    text = str(gpu_name or "").upper().replace(" ", "")
    return any(token in text for token in ("RTX50", "RTX5060", "RTX5070", "RTX5080", "RTX5090"))


def recommend_cuda_index(
    capability: tuple[int, int] | None = None,
    nvcc_version: str | None = None,
    gpu_name: str | None = None,
) -> str:
    # RTX 50 / Blackwell 是 sm_120，需要 CUDA 12.8 版 PyTorch wheel。
    if (capability and capability >= (12, 0)) or looks_like_rtx50(gpu_name):
        return CUDA_INDEX_CU128

    if str(nvcc_version or "").startswith("11.8"):
        return "https://download.pytorch.org/whl/cu118"
    if str(nvcc_version or "").startswith("11"):
        return "https://download.pytorch.org/whl/cu117"
    return CUDA_INDEX_CU121


def inspect_torch_cuda(test_tensor: bool = True) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_installed": False,
        "cuda_available": False,
        "cuda_usable": False,
        "cuda_issue": None,
        "cuda_warnings": [],
    }

    try:
        import torch
    except ImportError:
        info["cuda_issue"] = "尚未安裝 PyTorch。"
        return info

    info["torch_installed"] = True
    info["torch_version"] = getattr(torch, "__version__", "")
    info["torch_cuda"] = getattr(torch.version, "cuda", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                capability = torch.cuda.get_device_capability(0)
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_count"] = torch.cuda.device_count()
                info["cuda_capability"] = list(capability)
                info["cuda_capability_label"] = capability_text(capability)

                runtime_version = version_tuple(info["torch_cuda"])
                if capability >= (12, 0) and runtime_version < (12, 8):
                    info["cuda_issue"] = (
                        f"{info['gpu_name']} 是 {capability_text(capability)}，需要 CUDA 12.8 版 PyTorch；"
                        f"目前安裝的是 CUDA {info['torch_cuda'] or '未知'}。"
                    )
                elif test_tensor:
                    try:
                        torch.empty(1, device="cuda")
                        torch.cuda.synchronize()
                        info["cuda_usable"] = True
                    except Exception as exc:
                        info["cuda_issue"] = f"CUDA 測試失敗：{exc}"
                else:
                    info["cuda_usable"] = True
            else:
                info["cuda_issue"] = None if is_macos() else "PyTorch 目前偵測不到可用 CUDA。"
        except Exception as exc:
            info["cuda_issue"] = f"CUDA 偵測失敗：{exc}"

        info["cuda_warnings"] = [str(item.message) for item in caught]

    if not info["cuda_usable"] and not info["cuda_issue"] and info["cuda_warnings"]:
        info["cuda_issue"] = info["cuda_warnings"][0]

    return info


def inspect_torch_mps(test_tensor: bool = True) -> dict[str, Any]:
    info: dict[str, Any] = {
        "mps_available": False,
        "mps_usable": False,
        "mps_issue": None,
    }

    try:
        import torch
    except ImportError:
        info["mps_issue"] = "尚未安裝 PyTorch。"
        return info

    backend = getattr(getattr(torch, "backends", None), "mps", None)
    if backend is None:
        info["mps_issue"] = "目前的 PyTorch 不支援 Apple GPU (MPS)。" if is_macos() else None
        return info

    try:
        info["mps_built"] = bool(backend.is_built())
        info["mps_available"] = bool(backend.is_available())
        if info["mps_available"]:
            if test_tensor:
                try:
                    torch.empty(1, device="mps")
                    info["mps_usable"] = True
                except Exception as exc:
                    info["mps_issue"] = f"MPS 測試失敗：{exc}"
            else:
                info["mps_usable"] = True
        elif is_macos():
            info["mps_issue"] = "目前偵測不到可用的 Apple GPU (MPS)，將使用 CPU。"
    except Exception as exc:
        info["mps_issue"] = f"MPS 偵測失敗：{exc}"

    return info


def ffmpeg_binary_name() -> str:
    return "ffmpeg.exe" if is_windows() else "ffmpeg"


def ffmpeg_path_in_candidate(candidate: str) -> Path:
    return Path(candidate) / ffmpeg_binary_name()


def ffmpeg_candidates() -> list[str]:
    """中文註解：優先找 PATH，其次找各平台常見安裝位置。"""
    candidates: list[str] = []

    if is_windows():
        candidates.extend([r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin"])

        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            for app_name in ("CapCut", "JianyingPro"):
                app_root = Path(local_app_data) / app_name / "Apps"
                if app_root.is_dir():
                    for version_dir in sorted(app_root.iterdir(), reverse=True):
                        if version_dir.is_dir():
                            candidates.append(str(version_dir))
    elif is_macos():
        candidates.extend(
            [
                "/opt/homebrew/bin",
                "/usr/local/bin",
                "/opt/local/bin",
                "/usr/bin",
                "/Applications/CapCut.app/Contents/MacOS",
                "/Applications/CapCut.app/Contents/Frameworks",
            ]
        )
    else:
        candidates.extend(["/usr/local/bin", "/usr/bin", "/bin"])

    return list(dict.fromkeys(candidates))


def setup_ffmpeg() -> None:
    """中文註解：若系統 PATH 沒有 ffmpeg，就補上可找到的路徑。"""
    if shutil.which("ffmpeg"):
        return

    for candidate in ffmpeg_candidates():
        ffmpeg_path = ffmpeg_path_in_candidate(candidate)
        if ffmpeg_path.is_file():
            os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
            print(f"[ffmpeg] 使用路徑：{candidate}")
            return

    print("[ffmpeg] 找不到 ffmpeg，轉錄前請先安裝或確認 PATH。")


def detect_device() -> tuple[str, bool]:
    """中文註解：自動偵測 CUDA / Apple MPS，讓首次啟動就能決定預設運算裝置。"""
    cuda_info = inspect_torch_cuda()
    if cuda_info.get("cuda_usable"):
        name = cuda_info.get("gpu_name", "NVIDIA GPU")
        print(f"[GPU] 偵測到 {name}，CUDA 測試通過，預設使用 GPU。")
        return "cuda", True

    issue = cuda_info.get("cuda_issue")
    if issue:
        print(f"[GPU] {issue}")

    mps_info = inspect_torch_mps()
    if mps_info.get("mps_usable"):
        if mps_requested():
            print("[GPU] 偵測到 Apple GPU (MPS)，依 WHISPER_ENABLE_MPS=1 使用 MPS。")
            return "mps", False
        print("[GPU] 偵測到 Apple GPU (MPS)，但 Whisper 在 MPS 可能不穩定，預設使用 CPU。")
        return "cpu", False

    mps_issue = mps_info.get("mps_issue")
    if mps_issue:
        print(f"[GPU] {mps_issue}")

    print("[CPU] 未偵測到可用 GPU，改用 CPU。")
    return "cpu", False


def get_whisper_model():
    """中文註解：Whisper 模型採 lazy loading，避免一開頁面就佔用大量記憶體。"""
    global _whisper_model

    with _model_lock:
        if _whisper_model is None:
            import whisper

            print(f"[Whisper] 載入 medium 模型（device={DEVICE}）...")
            _whisper_model = whisper.load_model("medium", device=DEVICE)
            print("[Whisper] 模型載入完成。")

    return _whisper_model


def clear_torch_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def switch_runtime_device(target: str) -> None:
    global DEVICE, USE_FP16, _whisper_model

    with _model_lock:
        _whisper_model = None
        DEVICE = target
        USE_FP16 = target == "cuda"

    gc.collect()
    clear_torch_cache()


def looks_like_mps_backend_error(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in MPS_ERROR_MARKERS)


def update_job_progress(job_id: str, progress: int | None = None, text: str | None = None) -> None:
    with jobs_lock:
        current = jobs.get(job_id)
        if not current or current.status != "processing":
            return

        if progress is not None:
            current.progress = max(0, min(100, int(progress)))
        if text:
            current.progress_text = text


class WhisperProgress:
    def __init__(self, job_id: str, *args: Any, **kwargs: Any) -> None:
        self.job_id = job_id
        self.total = int(kwargs.get("total") or (args[0] if args else 0) or 0)
        self.n = 0
        update_job_progress(job_id, 12, "正在分析音訊內容")

    def __enter__(self) -> "WhisperProgress":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def update(self, amount: int | float = 1) -> None:
        self.n += int(amount or 0)
        if self.total <= 0:
            return

        ratio = max(0.0, min(1.0, self.n / self.total))
        progress = 12 + round(ratio * 78)
        update_job_progress(self.job_id, progress, f"正在轉錄語音：{progress}%")

    def close(self) -> None:
        return

    def set_description(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def write(self, *_args: Any, **_kwargs: Any) -> None:
        return


def transcribe_file(job_id: str, file_path: str) -> dict[str, Any]:
    update_job_progress(job_id, 5, "正在載入 Whisper 模型")
    model = get_whisper_model()
    update_job_progress(job_id, 10, "正在準備轉錄")

    # openai-whisper 內部用 tqdm 更新音訊 frame 進度；這裡接住它轉成前端百分比。
    with _transcribe_lock:
        transcribe_module = importlib.import_module("whisper.transcribe")
        original_tqdm = transcribe_module.tqdm.tqdm
        transcribe_module.tqdm.tqdm = lambda *args, **kwargs: WhisperProgress(job_id, *args, **kwargs)
        try:
            return model.transcribe(
                file_path,
                language=None,
                task="transcribe",
                fp16=USE_FP16,
                verbose=False,
            )
        finally:
            transcribe_module.tqdm.tqdm = original_tqdm


def fmt_time(seconds: float) -> str:
    milliseconds = int(round((seconds % 1) * 1000))
    whole_seconds = int(seconds)
    sec = whole_seconds % 60
    minute = (whole_seconds // 60) % 60
    hour = whole_seconds // 3600
    return f"{hour:02d}:{minute:02d}:{sec:02d},{milliseconds:03d}"


def merge_segments(segments: list[dict[str, Any]], mode: str = "standard") -> list[dict[str, Any]]:
    """中文註解：把 Whisper 的細碎片段合併成更好閱讀的字幕段落。"""
    if mode == "fine" or not segments:
        return segments

    if mode == "coarse":
        max_chars, max_sec, max_gap = 70, 10.0, 2.0
    else:
        max_chars, max_sec, max_gap = 40, 6.0, 1.5

    break_punct = {"。", "！", "？", ".", "!", "?", "，", ","}
    merged: list[dict[str, Any]] = []
    current = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": str(segments[0]["text"]).strip(),
    }

    for segment in segments[1:]:
        text = str(segment["text"]).strip()
        if not text:
            continue

        gap = float(segment["start"]) - float(current["end"])
        combined = (str(current["text"]) + " " + text).strip()
        combined_duration = float(segment["end"]) - float(current["start"])
        ends_break = bool(current["text"]) and str(current["text"])[-1] in break_punct

        can_merge = (
            gap <= max_gap
            and len(combined) <= max_chars
            and combined_duration <= max_sec
            and not ends_break
        )

        if can_merge:
            current["text"] = combined
            current["end"] = segment["end"]
        else:
            merged.append(current)
            current = {"start": segment["start"], "end": segment["end"], "text": text}

    merged.append(current)
    return merged


def segments_to_srt(segments: list[dict[str, Any]], mode: str = "standard") -> str:
    merged = merge_segments(segments, mode)
    blocks: list[str] = []

    for idx, segment in enumerate(merged, start=1):
        text = str(segment["text"]).strip()
        if not text:
            continue

        blocks.append(
            f"{idx}\n"
            f"{fmt_time(float(segment['start']))} --> {fmt_time(float(segment['end']))}\n"
            f"{text}\n"
        )

    return "\n".join(blocks)


def append_install_line(install_id: str, line: str) -> None:
    with install_lock:
        install_jobs[install_id].lines.append(line)


def set_install_status(install_id: str, status: str, message: str | None = None) -> None:
    with install_lock:
        install_jobs[install_id].status = status
        if message:
            install_jobs[install_id].lines.append("")
            install_jobs[install_id].lines.append(message)


def find_python_for_frontend() -> str:
    return sys.executable


def run_install_command(
    install_id: str,
    command: list[str],
    success_message: str,
    finish_success: bool = True,
) -> bool:
    append_install_line(install_id, f">>> {' '.join(command)}")
    append_install_line(install_id, "安裝中，首次下載可能需要一段時間，請耐心等待。")
    append_install_line(install_id, "")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                append_install_line(install_id, line)

        process.wait()
        if process.returncode == 0:
            if finish_success:
                set_install_status(install_id, "done", success_message)
            return True
        else:
            set_install_status(install_id, "error", f"安裝失敗，錯誤代碼：{process.returncode}")
            return False
    except Exception as exc:
        set_install_status(install_id, "error", f"安裝時發生例外：{exc}")
        return False


def run_cuda_torch_install(install_id: str, command: list[str]) -> None:
    if not run_install_command(install_id, command, f"CUDA 版 PyTorch 安裝完成，請重新啟動 {restart_entry_name()}。", False):
        return

    append_install_line(install_id, "")
    append_install_line(install_id, "正在確認 PyTorch CUDA 狀態...")
    verify_command = [
        sys.executable,
        "-c",
        (
            "import torch; "
            "print('PyTorch 版本：' + str(torch.__version__)); "
            "print('PyTorch 內建 CUDA：' + str(torch.version.cuda)); "
            "print('CUDA 可用：' + str(torch.cuda.is_available())); "
            "cc=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0); "
            "print('顯卡架構：sm_%d%d' % cc); "
            "cv=tuple(int(x) for x in (torch.version.cuda or '0.0').split('.')[:2]); "
            "ok=torch.cuda.is_available() and not (cc >= (12,0) and cv < (12,8)); "
            "torch.empty(1, device='cuda') if ok else None; "
            "torch.cuda.synchronize() if ok else None; "
            "raise SystemExit(0 if ok else 2)"
        ),
    ]

    try:
        verifier = subprocess.run(
            verify_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if verifier.stdout:
            for raw_line in verifier.stdout.splitlines():
                if raw_line.strip():
                    append_install_line(install_id, raw_line.strip())
        if verifier.returncode != 0:
            set_install_status(
                install_id,
                "error",
                "PyTorch 已安裝，但目前 CUDA 測試未通過。RTX 50 系列請安裝 CUDA 12.8 版 PyTorch；若仍失敗，請更新 NVIDIA 驅動或先改用 CPU。",
            )
            return
    except Exception as exc:
        set_install_status(install_id, "error", f"CUDA 驗證時發生例外：{exc}")
        return

    set_install_status(install_id, "done", f"CUDA 版 PyTorch 安裝工作完成，請重新啟動 {restart_entry_name()}。")


def run_whisper(job_id: str, file_path: str, seg_mode: str) -> None:
    try:
        with jobs_lock:
            current = jobs.get(job_id)
            if not current or current.status == "cancelled":
                return

        print(f"[Job {job_id[:8]}] 開始轉錄：{file_path}（device={DEVICE}）")
        try:
            result = transcribe_file(job_id, file_path)
        except Exception as exc:
            if DEVICE != "mps" or not looks_like_mps_backend_error(exc):
                raise

            print(f"[Job {job_id[:8]}] MPS 轉錄失敗，改用 CPU 重試：{exc}")
            update_job_progress(job_id, 8, "Apple MPS 不支援此轉錄流程，正在改用 CPU 重試")
            switch_runtime_device("cpu")
            print(f"[Job {job_id[:8]}] 開始 CPU 重試：{file_path}")
            result = transcribe_file(job_id, file_path)

        update_job_progress(job_id, 94, "正在產生 SRT 字幕")
        srt_content = segments_to_srt(result["segments"], seg_mode)
        with jobs_lock:
            current = jobs.get(job_id)
            if current and current.status != "cancelled":
                current.status = "done"
                current.srt = srt_content
                current.progress = 100
                current.progress_text = "字幕產生完成"

        print(f"[Job {job_id[:8]}] 轉錄完成。")
    except Exception as exc:
        with jobs_lock:
            current = jobs.get(job_id)
            if current:
                current.status = "error"
                current.error_msg = str(exc)
                current.progress_text = "轉錄失敗"

        print(f"[Job {job_id[:8]}] 轉錄失敗：{exc}")
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass


def build_env_check() -> dict[str, Any]:
    python_blocker = python_support_blocker()
    results: dict[str, Any] = {
        "python": {
            "ok": python_blocker is None,
            "label": "Python",
            "version": sys.version.split()[0],
            "note": find_python_for_frontend() if python_blocker is None else f"{find_python_for_frontend()}｜{python_blocker}",
        }
    }

    try:
        import flask

        results["flask"] = {
            "ok": True,
            "label": "Flask",
            "version": getattr(flask, "__version__", ""),
            "note": "",
        }
    except ImportError:
        results["flask"] = {"ok": False, "label": "Flask", "version": None, "note": "尚未安裝"}

    try:
        import whisper

        results["whisper"] = {
            "ok": True,
            "label": "openai-whisper",
            "version": getattr(whisper, "__version__", ""),
            "note": "",
        }
    except ImportError:
        results["whisper"] = {"ok": False, "label": "openai-whisper", "version": None, "note": "尚未安裝"}

    torch_info = inspect_torch_cuda()
    mps_info = inspect_torch_mps(test_tensor=False)
    if torch_info.get("torch_installed"):
        if torch_info.get("cuda_usable"):
            torch_note = "CUDA 可用"
        elif mps_info.get("mps_usable"):
            torch_note = "Apple GPU (MPS) 可用，但 Whisper 目前預設使用 CPU 較穩定"
        elif torch_info.get("cuda_available"):
            torch_note = torch_info.get("cuda_issue") or "偵測到 GPU，但 PyTorch CUDA 版本不相容"
        else:
            torch_note = "目前使用 CPU"
        results["torch"] = {
            "ok": True,
            "label": "PyTorch",
            "version": torch_info.get("torch_version", ""),
            "note": torch_note,
        }
    else:
        results["torch"] = {"ok": False, "label": "PyTorch", "version": None, "note": "尚未安裝"}

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        results["ffmpeg"] = {"ok": True, "label": "ffmpeg", "version": None, "note": ffmpeg_path}
    else:
        discovered = None
        for candidate in ffmpeg_candidates():
            if ffmpeg_path_in_candidate(candidate).is_file():
                discovered = candidate
                break

        if discovered:
            results["ffmpeg"] = {
                "ok": True,
                "label": "ffmpeg",
                "version": None,
                "note": f"已找到可用路徑：{discovered}",
            }
        else:
            ffmpeg_note = (
                "尚未找到 ffmpeg；macOS 可用 Homebrew 安裝：brew install ffmpeg。"
                if is_macos()
                else "尚未找到 ffmpeg，請參考 README 或用 CapCut 內建版本。"
            )
            results["ffmpeg"] = {
                "ok": False,
                "label": "ffmpeg",
                "version": None,
                "note": ffmpeg_note,
            }

    missing_pip = []
    if not results["flask"]["ok"]:
        missing_pip.append("flask")
    if not results["whisper"]["ok"]:
        missing_pip.append("openai-whisper")
    results["missing_pip"] = missing_pip
    results["python_supported"] = python_blocker is None
    results["python_blocker"] = python_blocker
    results["platform"] = platform_label()
    results["restart_command"] = restart_entry_name()
    return results


setup_ffmpeg()
DEVICE, USE_FP16 = detect_device()


@app.route("/")
def index():
    response = send_file(INDEX_FILE)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "請先選擇音訊或影片檔。"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"error": f"不支援的檔案格式：{ext}。請使用 {allowed}。"}), 400

    seg_mode = request.form.get("seg_mode", "standard")
    if seg_mode not in {"fine", "standard", "coarse"}:
        seg_mode = "standard"

    job_id = str(uuid.uuid4())
    save_path = str(UPLOAD_DIR / f"{job_id}.{ext}")
    file.save(save_path)

    with jobs_lock:
        jobs[job_id] = JobState(
            status="processing",
            filename=file.filename,
            file_path=save_path,
            progress=2,
            progress_text="檔案已上傳，正在排入轉錄",
        )

    thread = threading.Thread(target=run_whisper, args=(job_id, save_path, seg_mode), daemon=True)
    thread.start()

    print(f"[Upload] 建立工作 {job_id[:8]}，檔案：{file.filename}，模式：{seg_mode}")
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"status": "not_found"}), 404

    payload: dict[str, Any] = {
        "status": job.status,
        "progress": job.progress,
        "progress_text": job.progress_text,
    }
    if job.status == "done":
        payload["srt"] = job.srt
        payload["filename"] = Path(job.filename).stem + ".srt"
    elif job.status == "error":
        payload["error_msg"] = job.error_msg

    return jsonify(payload)


@app.route("/download/<job_id>")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job or job.status != "done" or not job.srt:
        return "找不到可下載的字幕檔。", 404

    filename = Path(job.filename).stem + ".srt"
    buffer = BytesIO(job.srt.encode("utf-8-sig"))
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job and job.status == "processing":
            job.status = "cancelled"
            job.error_msg = "使用者已取消轉錄。"

    return jsonify({"ok": True})


@app.route("/unload-model", methods=["POST"])
def unload_model():
    global _whisper_model

    freed = False
    with _model_lock:
        if _whisper_model is not None:
            _whisper_model = None
            freed = True

    gc.collect()

    message = "目前沒有已載入的模型。"
    if freed:
        clear_torch_cache()
        if DEVICE == "cuda":
            message = "模型已卸載，GPU VRAM 已釋放。"
        elif DEVICE == "mps":
            message = "模型已卸載，Apple GPU 快取已釋放。"
        else:
            message = "模型已卸載，記憶體已釋放。"

    print(f"[Unload] {message}")
    return jsonify({"ok": True, "msg": message})


@app.route("/device-info")
def device_info():
    info: dict[str, Any] = {
        "device": DEVICE,
        "fp16": USE_FP16,
        "model_loaded": _whisper_model is not None,
        "mps_experimental": True,
        "mps_enabled_by_env": mps_requested(),
    }

    cuda_info = inspect_torch_cuda(test_tensor=False)
    mps_info = inspect_torch_mps(test_tensor=False)
    info.update(cuda_info)
    info.update(mps_info)

    try:
        import torch

        if cuda_info.get("cuda_available"):
            props = torch.cuda.get_device_properties(0)
            info["gpu_total"] = round(props.total_memory / 1024**3, 1)
            info["gpu_used"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
    except Exception:
        pass

    return jsonify(info)


@app.route("/set-device", methods=["POST"])
def set_device():
    global DEVICE, USE_FP16, _whisper_model

    target = (request.get_json(silent=True) or {}).get("device", "cpu")
    if target not in {"cpu", "cuda", "mps"}:
        return jsonify({"error": "device 必須為 cpu、cuda 或 mps。"}), 400

    try:
        import torch
    except ImportError:
        return jsonify({"error": "找不到 torch，請先安裝 PyTorch。"}), 500

    if target == "cuda":
        cuda_info = inspect_torch_cuda()
        if not cuda_info.get("cuda_usable"):
            return jsonify({"error": cuda_info.get("cuda_issue") or "目前偵測不到可用的 CUDA GPU。"}), 400
    elif target == "mps":
        mps_info = inspect_torch_mps()
        if not mps_info.get("mps_usable"):
            return jsonify({"error": mps_info.get("mps_issue") or "目前偵測不到可用的 Apple GPU (MPS)。"}), 400

    switch_runtime_device(target)
    print(f"[Device] 切換到 {DEVICE.upper()}。")
    return jsonify({"ok": True, "device": DEVICE})


@app.route("/cuda-diagnose")
def cuda_diagnose():
    python_blocker = python_support_blocker()
    cuda_info = inspect_torch_cuda()
    mps_info = inspect_torch_mps()
    cuda_install_blocker = python_blocker
    if is_macos() and not cuda_install_blocker:
        cuda_install_blocker = "macOS 不使用 CUDA 版 PyTorch；Apple Silicon 可使用 MPS，其他 Mac 會使用 CPU。"

    result: dict[str, Any] = {
        "platform": platform_label(),
        "python_version": python_version_text(),
        "python_executable": find_python_for_frontend(),
        "python_supported": python_blocker is None,
        "cuda_install_supported": cuda_install_blocker is None,
        "cuda_install_blocker": cuda_install_blocker,
        "mps_experimental": True,
        "mps_enabled_by_env": mps_requested(),
        "restart_command": restart_entry_name(),
    }
    result.update(cuda_info)
    result.update(mps_info)

    if shutil.which("nvcc"):
        try:
            output = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.STDOUT)
            import re

            match = re.search(r"release\s+([\d.]+)", output)
            result["nvcc_version"] = match.group(1) if match else output.strip().splitlines()[-1]
        except Exception as exc:
            result["nvcc_version"] = str(exc)
    else:
        result["nvcc_version"] = None

    if shutil.which("nvidia-smi"):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.STDOUT,
            )
            result["nvidia_smi"] = output.strip()
            if output.strip() and not result.get("gpu_name"):
                result["gpu_name"] = output.strip().splitlines()[0].split(",", 1)[0].strip()
        except Exception:
            result["nvidia_smi"] = None
    else:
        result["nvidia_smi"] = None

    if is_macos():
        result["recommended_index"] = None
    else:
        capability = tuple(result["cuda_capability"]) if result.get("cuda_capability") else None
        result["recommended_index"] = recommend_cuda_index(capability, result.get("nvcc_version"), result.get("gpu_name") or result.get("nvidia_smi"))
    return jsonify(result)


@app.route("/install-cuda-torch", methods=["POST"])
def install_cuda_torch():
    data = request.get_json(silent=True) or {}
    index_url = data.get("index_url", CUDA_INDEX_CU128)

    if is_macos():
        return jsonify({"error": "macOS 不需要安裝 CUDA 版 PyTorch；請使用一般 PyTorch，Apple Silicon 會走 MPS 或 CPU。"}), 400

    python_blocker = python_support_blocker()
    if python_blocker:
        return jsonify({"error": python_blocker}), 400

    install_id = str(uuid.uuid4())
    with install_lock:
        install_jobs[install_id] = InstallJobState()

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "--index-url",
        index_url,
        "--upgrade",
        "--force-reinstall",
        "--progress-bar",
        "off",
    ]

    thread = threading.Thread(
        target=run_cuda_torch_install,
        args=(install_id, command),
        daemon=True,
    )
    thread.start()
    return jsonify({"install_id": install_id})


@app.route("/env-check")
def env_check():
    return jsonify(build_env_check())


@app.route("/install", methods=["POST"])
def install_packages():
    data = request.get_json(silent=True) or {}
    packages = data.get("packages", ["flask", "openai-whisper"])
    if not isinstance(packages, list) or not packages:
        packages = ["flask", "openai-whisper"]
    packages = [str(package) for package in packages]

    python_blocker = python_support_blocker()
    if python_blocker and packages_need_pytorch(packages):
        return jsonify({"error": python_blocker}), 400

    install_id = str(uuid.uuid4())
    with install_lock:
        install_jobs[install_id] = InstallJobState()

    command = [sys.executable, "-m", "pip", "install", *packages, "--progress-bar", "off"]
    thread = threading.Thread(
        target=run_install_command,
        args=(install_id, command, f"套件安裝完成，請重新執行 {restart_entry_name()}。"),
        daemon=True,
    )
    thread.start()
    return jsonify({"install_id": install_id})


@app.route("/install-status/<install_id>")
def install_status(install_id: str):
    with install_lock:
        job = install_jobs.get(install_id)

    if not job:
        return jsonify({"status": "not_found"}), 404

    return jsonify({"status": job.status, "lines": job.lines})


if __name__ == "__main__":
    port = server_port()
    print("=" * 42)
    print("  Whisper 字幕神器啟動中")
    print(f"  服務位置：http://localhost:{port}")
    print("=" * 42)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
