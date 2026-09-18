import os
import subprocess
import sys
import wave
from typing import Any, Dict, Optional

import yt_dlp
from yt_dlp.utils import YoutubeDLError


class DownloadError(Exception):
    """Custom exception raised when video downloading or audio extraction fails."""
    pass


def download_video(url: str, output_dir: str) -> Dict[str, Any]:
    """
    Downloads the best-quality video from a given URL (e.g. YouTube) and
    extracts a 16kHz mono WAV audio track suitable for speech recognition (Whisper).

    Args:
        url: The video URL to download.
        output_dir: The directory where the downloaded video and extracted audio will be stored.

    Returns:
        A dictionary containing:
            - "video_path": Absolute path to the downloaded video file (.mp4).
            - "audio_path": Absolute path to the extracted 16kHz mono WAV file.
            - "duration": Duration of the video/audio in seconds (float).

    Raises:
        DownloadError: If the URL is invalid, video is unavailable/private, network fails,
                       or audio extraction fails.
    """
    if not url or not isinstance(url, str) or not url.strip():
        raise DownloadError("A valid, non-empty URL string must be provided.")

    url = url.strip()
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    last_percent = -1
    current_download_filename: Optional[str] = None

    def _progress_hook(d: Dict[str, Any]) -> None:
        nonlocal last_percent, current_download_filename

        status = d.get("status")
        filename = d.get("filename")

        # Reset percentage tracking if yt-dlp switches to downloading another component (e.g. audio stream)
        if filename and filename != current_download_filename:
            current_download_filename = filename
            last_percent = -1

        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            if total and total > 0:
                percent = int((downloaded / total) * 100)
                if percent != last_percent:
                    last_percent = percent
                    print(f"[download] {percent}% of video...", flush=True)
        elif status == "finished":
            if last_percent != 100:
                last_percent = 100
                print("[download] 100% of video...", flush=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best",
        "outtmpl": os.path.join(abs_output_dir, "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [_progress_hook],
        "js_runtimes": {"node": {}},
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except YoutubeDLError as err:
        raise DownloadError(f"Failed to download video from '{url}': {err}") from err
    except Exception as err:
        raise DownloadError(f"Unexpected error while downloading '{url}': {err}") from err

    if not info:
        raise DownloadError(f"No video information could be retrieved for URL: {url}")

    video_id = info.get("id")
    if not video_id:
        raise DownloadError(f"Could not determine video ID from URL: {url}")

    # Determine video file path
    video_path: Optional[str] = None
    reqs = info.get("requested_downloads")
    if reqs and isinstance(reqs, list) and len(reqs) > 0:
        candidate = reqs[0].get("filepath")
        if candidate and os.path.exists(candidate):
            video_path = os.path.abspath(candidate)

    if not video_path:
        default_target = os.path.join(abs_output_dir, f"{video_id}.mp4")
        if os.path.exists(default_target):
            video_path = default_target
        else:
            # Check for any file matching video_id in output directory
            candidates = [
                os.path.join(abs_output_dir, f)
                for f in os.listdir(abs_output_dir)
                if f.startswith(f"{video_id}.") and not f.endswith(".wav")
            ]
            if candidates:
                video_path = candidates[0]

    if not video_path or not os.path.exists(video_path):
        raise DownloadError(f"Downloaded video file not found in output directory: {abs_output_dir}")

    if os.path.getsize(video_path) == 0:
        raise DownloadError(f"Downloaded video file is empty: {video_path}")

    # Extract 16kHz mono WAV audio track suitable for Whisper
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]

    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as err:
        raise DownloadError(
            "ffmpeg executable not found in system PATH. Please ensure ffmpeg is installed."
        ) from err
    except subprocess.CalledProcessError as err:
        error_msg = err.stderr.strip() if err.stderr else str(err)
        raise DownloadError(f"Failed to extract 16kHz mono audio using ffmpeg: {error_msg}") from err

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise DownloadError(f"Extracted audio file is missing or empty: {audio_path}")

    # Determine duration
    duration = float(info.get("duration") or 0.0)
    if duration <= 0.0:
        try:
            with wave.open(audio_path, "rb") as wf:
                n_frames = wf.getnframes()
                frame_rate = wf.getframerate()
                if frame_rate > 0:
                    duration = float(n_frames / frame_rate)
        except Exception:
            duration = 0.0

    return {
        "video_path": os.path.abspath(video_path),
        "audio_path": os.path.abspath(audio_path),
        "duration": float(duration),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python downloader.py <video_url> [output_dir]")
        sys.exit(1)

    target_url = sys.argv[1]
    out_directory = sys.argv[2] if len(sys.argv) > 2 else os.path.join("samples", "input")

    try:
        print(f"Downloading from: {target_url}")
        res = download_video(target_url, out_directory)
        print("\nDownload complete:")
        print(f"  Video Path: {res['video_path']}")
        print(f"  Audio Path: {res['audio_path']}")
        print(f"  Duration:   {res['duration']:.2f}s")
    except DownloadError as e:
        print(f"\n[DownloadError]: {e}", file=sys.stderr)
        sys.exit(1)
