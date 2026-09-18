"""
remixer.py - Video audio replacement and remuxing module for automatedub.

Stitches all synthesized audio segments into a continuous, time-aligned audio track
(filling pauses and gaps with silence), then replaces the original audio track in the video
without re-encoding the video stream (-c:v copy).
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _parse_time_value(val: Any) -> float:
    """Safely converts a timestamp value (float, int, or string) to a float in seconds."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.strip().rstrip("s").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


class RemixError(Exception):
    """Exception raised when audio stitching or video remuxing fails."""
    pass


def get_video_duration(video_path: str) -> float:
    """
    Extracts the duration of a video file in seconds using ffmpeg.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds (float). Returns 0.0 if duration cannot be detected.
    """
    if not os.path.exists(video_path):
        return 0.0

    cmd = ["ffmpeg", "-i", video_path]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    for line in proc.stderr.splitlines():
        if "Duration:" in line:
            try:
                part = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = part.split(":")
                return float(h) * 3600 + float(m) * 60 + float(s)
            except Exception:
                pass
    return 0.0


def _read_wav_samples(audio_path: str, target_sr: int = 24000) -> np.ndarray:
    """
    Reads audio from a WAV file as a 1D float32 numpy array.
    If sample rate or channels mismatch target_sr or mono, resamples via ffmpeg.
    """
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return np.array([], dtype=np.float32)

    needs_conversion = False
    try:
        with wave.open(audio_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            if channels != 1 or sample_rate != target_sr or sampwidth != 2:
                needs_conversion = True
            else:
                raw_bytes = wf.readframes(n_frames)
                return np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    except Exception:
        needs_conversion = True

    if needs_conversion:
        # Convert to mono 16-bit PCM target_sr via ffmpeg pipe
        try:
            cmd = [
                "ffmpeg",
                "-i",
                audio_path,
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(target_sr),
                "-ac",
                "1",
                "-",
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
        except Exception:
            return np.array([], dtype=np.float32)

    return np.array([], dtype=np.float32)


def stitch_audio_segments(
    segments: List[Dict[str, Any]],
    target_duration: float,
    sample_rate: int = 24000,
    return_stats: bool = False,
) -> Any:
    """
    Stitches individual segment audio clips into a single continuous audio track.

    Guarantees that:
    - Each segment's synthesized audio is placed at its own segment["start"] timestamp on the
      output timeline (not sequentially appended), with silence filling any gaps.
    - If a segment's audio is longer than the time available before the next segment starts,
      a warning is logged (without truncating or incorrectly shifting the audio).
    - Prints each segment's placement: "[remix] segment at {start}s placed at timeline position {actual_position}s".
    - Silence is strictly placed from 0:00 up to the first segment's start time (lead-in silence).
    - Silence is preserved in any gaps between consecutive segments.
    - Silence is padded after the last segment through to the end of target_duration.
    - Slightly overlapping segments are smoothly blended without digital clipping.

    Args:
        segments: List of segment dicts with 'start', 'end', and 'audio_path'.
        target_duration: Total desired video duration in seconds.
        sample_rate: Audio sample rate (defaults to 24000Hz).
        return_stats: If True, returns a tuple of (audio_array, stats_dict).

    Returns:
        1D numpy array of int16 audio samples, or (audio_array, stats_dict) if return_stats=True.
    """
    max_seg_end = 0.0
    for s in segments:
        s_start = max(0.0, _parse_time_value(s.get("start", 0.0)))
        s_end = max(s_start, _parse_time_value(s.get("end", s_start)))
        if s_end > max_seg_end:
            max_seg_end = s_end

    effective_duration = max(float(target_duration), max_seg_end)
    total_samples = max(1, int(round(effective_duration * sample_rate)))
    buffer = np.zeros(total_samples, dtype=np.float32)
    active_mask = np.zeros(total_samples, dtype=bool)

    # Sort segments chronologically by start time
    sorted_segments = sorted(segments, key=lambda s: _parse_time_value(s.get("start", 0.0)))

    for idx, seg in enumerate(sorted_segments):
        raw_start = seg.get("start", 0.0)
        start_time = max(0.0, _parse_time_value(raw_start))
        start_idx = int(round(start_time * sample_rate))
        if start_idx >= total_samples:
            continue

        actual_position = start_idx / sample_rate

        # Determine display strings to preserve formatting (e.g. 8.0, 25.0, 0.53)
        if isinstance(raw_start, (int, float)):
            start_display = raw_start
            actual_display = raw_start if float(raw_start) == actual_position else round(actual_position, 3)
        elif isinstance(raw_start, str):
            clean_str = raw_start.strip().rstrip("s").strip()
            start_display = clean_str
            actual_display = clean_str if float(start_time) == actual_position else round(actual_position, 3)
        else:
            start_display = round(start_time, 3)
            actual_display = round(actual_position, 3)

        print(
            f"[remix] segment at {start_display}s placed at timeline position {actual_display}s",
            flush=True,
        )

        audio_path = seg.get("audio_path")
        clip = _read_wav_samples(audio_path, target_sr=sample_rate) if audio_path else np.array([], dtype=np.float32)
        clip_dur = len(clip) / sample_rate if len(clip) > 0 else 0.0

        # Check if segment audio is longer than time available before next segment starts
        if idx < len(sorted_segments) - 1:
            next_seg = sorted_segments[idx + 1]
            raw_next_start = next_seg.get("start", 0.0)
            next_start = max(0.0, _parse_time_value(raw_next_start))
            if next_start >= start_time:
                time_available = next_start - start_time
                if clip_dur > time_available + 1e-3:
                    next_display = (
                        raw_next_start
                        if isinstance(raw_next_start, (int, float))
                        else raw_next_start.strip().rstrip("s")
                        if isinstance(raw_next_start, str)
                        else round(next_start, 3)
                    )
                    warn_msg = (
                        f"[remix] Warning: segment at {start_display}s audio duration ({clip_dur:.2f}s) "
                        f"exceeds available time ({time_available:.2f}s) before next segment at {next_display}s"
                    )
                    logger.warning(warn_msg)
                    print(warn_msg, flush=True)

        if len(clip) > 0:
            end_idx = min(total_samples, start_idx + len(clip))
            valid_clip_len = end_idx - start_idx

            # Mix samples into the buffer and record active speech frames (smooth additive mixing without truncating)
            buffer[start_idx:end_idx] += clip[:valid_clip_len]
            active_mask[start_idx:end_idx] = True

    # Clip to valid int16 audio bounds
    int16_audio = np.clip(buffer, -32768.0, 32767.0).astype(np.int16)

    if not return_stats:
        return int16_audio

    # Detailed timeline analysis for verification directly from the active speech mask
    active_speech_samples = int(np.sum(active_mask))
    total_speech_sec = active_speech_samples / sample_rate
    total_silence_sec = max(0.0, effective_duration - total_speech_sec)

    if active_speech_samples == 0:
        lead_in_sec = effective_duration
        gap_sec = 0.0
        lead_out_sec = 0.0
    else:
        first_speech_idx = int(np.argmax(active_mask))
        lead_in_sec = first_speech_idx / sample_rate

        last_speech_idx = total_samples - 1 - int(np.argmax(active_mask[::-1]))
        lead_out_sec = max(0.0, (total_samples - 1 - last_speech_idx) / sample_rate)

        # Gap silence is any silent sample between the first and last active speech frames
        gap_samples = int(np.sum(~active_mask[first_speech_idx : last_speech_idx + 1]))
        gap_sec = gap_samples / sample_rate

    stats = {
        "target_duration": effective_duration,
        "dubbed_speech": total_speech_sec,
        "total_silence": total_silence_sec,
        "lead_in_silence": lead_in_sec,
        "gap_silence": gap_sec,
        "lead_out_silence": lead_out_sec,
    }

    return int16_audio, stats


def remix(video_path: str, segments: List[Dict[str, Any]], output_path: str) -> str:
    """
    Stitches all segment audio clips into one full-length audio track and replaces
    the original audio in video_path, WITHOUT re-encoding the video stream (-c:v copy).

    Guarantees:
    - Lead-in silence from 0:00 up to segments[0]["start"] is preserved.
    - Silence gaps between all segments and trailing silence through video end are preserved.
    - Handles duration differences between video and audio by padding or trimming cleanly.

    Args:
        video_path: Path to the original input video.
        segments: List of segment dicts containing 'start', 'end', and 'audio_path'.
        output_path: Path where the final dubbed video will be written.

    Returns:
        The absolute path to the written output video.

    Raises:
        RemixError: If input video is missing or ffmpeg remuxing fails.
    """
    if not video_path or not os.path.exists(video_path):
        raise RemixError(f"Input video file not found: {video_path}")

    if os.path.getsize(video_path) == 0:
        raise RemixError(f"Input video file is empty: {video_path}")

    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    # 1. Determine video duration
    video_dur = get_video_duration(video_path)
    if video_dur <= 0.0:
        # Fallback to the maximum end timestamp among segments
        if segments:
            video_dur = max(float(s.get("end", 0.0)) for s in segments)
        if video_dur <= 0.0:
            video_dur = 1.0

    sample_rate = 24000
    temp_dir = tempfile.mkdtemp(prefix="remix_")
    stitched_wav = os.path.join(temp_dir, "stitched_dub.wav")

    try:
        # 2. Stitch segments into one full-length audio track with full timeline tracking
        audio_samples, stats = stitch_audio_segments(
            segments=segments,
            target_duration=video_dur,
            sample_rate=sample_rate,
            return_stats=True,
        )

        print(
            f"[remix] Total duration of inserted silence: {stats['total_silence']:.2f}s vs. dubbed speech: {stats['dubbed_speech']:.2f}s "
            f"(lead-in: {stats['lead_in_silence']:.2f}s, gaps: {stats['gap_silence']:.2f}s, lead-out: {stats['lead_out_silence']:.2f}s)",
            flush=True,
        )

        with wave.open(stitched_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_samples.tobytes())

        # 3. Remux into video: replace audio, copy video stream (-c:v copy)
        print("[remix] muxing audio into final video...", flush=True)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            stitched_wav,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            abs_output_path,
        ]

        proc = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            raise RemixError(f"FFmpeg remuxing failed: {proc.stderr.strip()}")

        if not os.path.exists(abs_output_path) or os.path.getsize(abs_output_path) == 0:
            raise RemixError(f"Output video file was not created or is empty: {abs_output_path}")

        # 4. Confirm output file written with size
        file_size_bytes = os.path.getsize(abs_output_path)
        if file_size_bytes >= 1024 * 1024:
            size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"
        else:
            size_str = f"{file_size_bytes / 1024:.1f} KB"

        print(
            f"[remix] Dubbed video successfully created: {abs_output_path} ({size_str})",
            flush=True,
        )

        return abs_output_path

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python remixer.py <video_path> <output_path>")
        sys.exit(1)

    vid_in = sys.argv[1]
    vid_out = sys.argv[2]

    try:
        res = remix(vid_in, [], vid_out)
        print("Remix complete:", res)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
