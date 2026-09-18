"""
transcriber.py - Speech transcription and translation module for automatedub.

Uses faster-whisper (CTranslate2-backed Whisper) for high-performance,
low-latency speech transcription and translation.
"""

import os
import sys
from typing import Any, Dict, List, Optional


def _setup_cuda_env() -> None:
    """Ensure CUDA runtime libraries from torch are discoverable for ctranslate2 on Windows."""
    if sys.platform == "win32":
        try:
            import torch

            if torch.cuda.is_available():
                lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
                if os.path.exists(lib_dir):
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(lib_dir)
                        except (FileNotFoundError, OSError):
                            pass
                    os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")

                    # Ensure cublas64_12 is available if torch provides cublas64_13
                    c13 = os.path.join(lib_dir, "cublas64_13.dll")
                    c12 = os.path.join(lib_dir, "cublas64_12.dll")
                    cL13 = os.path.join(lib_dir, "cublasLt64_13.dll")
                    cL12 = os.path.join(lib_dir, "cublasLt64_12.dll")

                    if os.path.exists(c13) and not os.path.exists(c12):
                        import shutil

                        shutil.copyfile(c13, c12)
                    if os.path.exists(cL13) and not os.path.exists(cL12):
                        import shutil

                        shutil.copyfile(cL13, cL12)

                    import ctypes

                    if os.path.exists(c12):
                        try:
                            ctypes.CDLL(c12)
                        except Exception:
                            pass
                    if os.path.exists(cL12):
                        try:
                            ctypes.CDLL(cL12)
                        except Exception:
                            pass
        except Exception:
            pass


_setup_cuda_env()

import torch
from faster_whisper import WhisperModel


def _format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS string representation."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def transcribe(audio_path: str, model_size: str = "medium") -> List[Dict[str, Any]]:
    """
    Transcribe and translate an audio track into English segments using faster-whisper.

    Auto-detects the source language and uses Whisper's task="translate" mode in a single pass
    to produce English speech segments with accurate timestamps.

    Model Size Tradeoff (Speed vs. Accuracy):
    -----------------------------------------
    - 'tiny' / 'base':
        Fastest inference speed and lowest memory usage (~1 GB VRAM). Excellent for rapid
        testing and iterations, but has lower transcription accuracy on accented, noisy,
        or complex multi-speaker audio.
    - 'small':
        Good balance for clean audio and everyday speech; ~2 GB VRAM requirement.
    - 'medium' (Default):
        High translation quality, natural phrasing, and robust alignment across multiple
        source languages (German, French, Hindi, Spanish, etc.), while running efficiently
        on modern NVIDIA GPUs (~5 GB VRAM). Recommended for production dubbing pipelines.
    - 'large-v3':
        Highest translation quality and best handling of technical jargon and rare dialects,
        but has a higher VRAM footprint (~10 GB) and higher compute latency.

    Args:
        audio_path: Path to the 16kHz mono audio file (.wav).
        model_size: Whisper model size name ('tiny', 'base', 'small', 'medium', 'large-v3').
                    Defaults to 'medium'.

    Returns:
        List of segment dictionaries matching the project data contract:
            [
                {
                    "start": float,        # Start time in seconds
                    "end": float,          # End time in seconds
                    "text": str,           # Original or transcribed segment text
                    "english_text": str,   # Translated English text
                    "lang": str            # Detected source language code (e.g. 'en', 'fr', 'hi')
                },
                ...
            ]

    Raises:
        FileNotFoundError: If the specified audio file does not exist.
        ValueError: If the audio file is empty.
    """
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if os.path.getsize(audio_path) == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"[transcribe] Using device: {device}", flush=True)
    print(f"[transcribe] Loading Whisper model '{model_size}' ({compute_type})...", flush=True)

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"[transcribe] Processing audio: {audio_path}...", flush=True)
    segments_gen, info = model.transcribe(
        audio_path,
        task="translate",
        beam_size=5,
        vad_filter=True,
    )

    detected_lang = info.language
    lang_prob = info.language_probability
    print(
        f"[transcribe] Detected source language: '{detected_lang}' (probability: {lang_prob:.1%})",
        flush=True,
    )

    results: List[Dict[str, Any]] = []

    for seg in segments_gen:
        text_content = seg.text.strip()
        if not text_content:
            continue

        start_time = round(float(seg.start), 3)
        end_time = round(float(seg.end), 3)

        segment_dict = {
            "start": start_time,
            "end": end_time,
            "text": text_content,
            "english_text": text_content,
            "lang": detected_lang,
        }
        results.append(segment_dict)

        start_ts = _format_timestamp(start_time)
        end_ts = _format_timestamp(end_time)
        print(f"[transcribe] {start_ts} -> {end_ts}: {text_content}", flush=True)

    print(f"[transcribe] Transcription complete. Generated {len(results)} segments.", flush=True)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_path> [model_size]")
        sys.exit(1)

    target_audio = sys.argv[1]
    m_size = sys.argv[2] if len(sys.argv) > 2 else "medium"

    try:
        segments = transcribe(target_audio, model_size=m_size)
        print(f"\nExtracted {len(segments)} segments successfully.")
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)
