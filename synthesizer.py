"""
synthesizer.py - Speech synthesis and duration synchronization module for automatedub.

Synthesizes natural English speech clips from translated segments using edge-tts,
estimates original speaker gender to match voices naturally, and synchronizes durations
without robotic pitch distortion using ffmpeg's atempo filter, silence padding, and trimming.
"""

import asyncio
import concurrent.futures
import logging
import os
import shutil
import subprocess
import sys
import wave
from typing import Any, Dict, List, Optional, Tuple

import edge_tts
import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Curated natural edge-tts voices per gender
VOICE_MALE = "en-US-ChristopherNeural"
VOICE_FEMALE = "en-US-JennyNeural"
VOICE_DEFAULT = "en-US-AriaNeural"


def _trim_trailing_silence(audio_path: str, threshold: int = 150) -> float:
    """
    Safely trims trailing digital silence from a 16-bit PCM mono WAV file without cutting speech.
    Modifies the file in place and returns the new duration in seconds.
    """
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return 0.0

    try:
        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)

        if n_channels != 1 or sampwidth != 2 or n_frames == 0:
            return _get_audio_duration(audio_path)

        samples = np.frombuffer(frames, dtype=np.int16)
        active_indices = np.where(np.abs(samples) > threshold)[0]
        if len(active_indices) == 0:
            return len(samples) / framerate

        last_active_idx = active_indices[-1]
        # Keep a subtle 50ms safety buffer after last active sample to prevent abrupt cutoffs
        pad_samples = int(round(0.05 * framerate))
        keep_samples = min(len(samples), last_active_idx + 1 + pad_samples)

        # Only modify file if more than 50ms of trailing silence can be removed
        if keep_samples < len(samples) - pad_samples:
            trimmed_samples = samples[:keep_samples]
            with wave.open(audio_path, "wb") as wf:
                wf.setnchannels(n_channels)
                wf.setsampwidth(sampwidth)
                wf.setframerate(framerate)
                wf.writeframes(trimmed_samples.tobytes())
            return len(trimmed_samples) / framerate

        return len(samples) / framerate
    except Exception:
        return _get_audio_duration(audio_path)


def _get_audio_duration(file_path: str) -> float:
    """Returns the duration of an audio file in seconds."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return 0.0

    # Try standard wave module first (for WAV files)
    try:
        with wave.open(file_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return float(frames / rate)
    except Exception:
        pass

    # Fallback to ffmpeg for other audio formats
    cmd = [
        "ffmpeg",
        "-i",
        file_path,
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        for line in proc.stderr.splitlines():
            if "Duration:" in line:
                part = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = part.split(":")
                return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        pass

    return 0.0


def _generate_silent_wav(output_path: str, duration: float, sample_rate: int = 24000) -> None:
    """Generates a silent mono WAV file of the given duration using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",
        "-t",
        f"{max(0.1, duration):.3f}",
        "-acodec",
        "pcm_s16le",
        output_path,
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _estimate_gender_and_confidence(audio_path: str) -> Tuple[str, float, float]:
    """
    Estimates speaker gender, confidence, and median fundamental frequency (F0) from audio.

    Uses librosa.pyin across speech frames.
    - avg_f0 < 150 Hz -> 'male'
    - avg_f0 > 185 Hz -> 'female'
    - 150 Hz <= avg_f0 <= 185 Hz -> 'unspecified' (ambiguous overlap range)

    Returns:
        Tuple of (gender_str, confidence_float, avg_f0_float).
    """
    if not audio_path or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return "unspecified", 0.50, 0.0

    try:
        # Load up to 60 seconds of audio at 16kHz
        y, sr = librosa.load(audio_path, sr=16000, duration=60.0)
        if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
            return "unspecified", 0.50, 0.0

        # Compute pitch via probabilistic YIN (pyin)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        valid_f0 = f0[~np.isnan(f0)]

        if len(valid_f0) < 5:
            return "unspecified", 0.50, 0.0

        # Focus on speech fundamental frequency range (65Hz - 350Hz) to filter out noise
        speech_f0 = valid_f0[valid_f0 <= 350.0]
        if len(speech_f0) >= 5:
            avg_f0 = float(np.median(speech_f0))
        else:
            avg_f0 = float(np.median(valid_f0))

        # Ambiguous overlap zone: 150 Hz to 185 Hz
        if avg_f0 < 150.0:
            gender = "male"
            diff = 150.0 - avg_f0
            confidence = min(0.95, 0.60 + (diff / 70.0) * 0.35)
        elif avg_f0 > 185.0:
            gender = "female"
            diff = avg_f0 - 185.0
            confidence = min(0.95, 0.60 + (diff / 70.0) * 0.35)
        else:
            gender = "unspecified"
            confidence = 0.50

        return gender, float(confidence), avg_f0

    except Exception:
        return "unspecified", 0.50, 0.0


def guess_gender(audio_path: str) -> str:
    """
    Estimates speaker gender from pitch using librosa.

    Calculates median fundamental frequency (F0) from the audio file.
    - below 150Hz -> 'male'
    - above 185Hz -> 'female'
    - 150Hz - 185Hz -> 'unspecified' (ambiguous overlap range)

    Args:
        audio_path: Path to the source audio file.

    Returns:
        'male', 'female', or 'unspecified'.
    """
    gender, _, _ = _estimate_gender_and_confidence(audio_path)
    return gender


async def _synthesize_text_async(text: str, voice: str, temp_mp3_path: str) -> None:
    """Asynchronously synthesizes speech using edge-tts."""
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(temp_mp3_path)


def _synthesize_text(text: str, voice: str, temp_mp3_path: str) -> None:
    """Synchronous wrapper for _synthesize_text_async, managing event loop state."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, _synthesize_text_async(text, voice, temp_mp3_path))
            future.result()
    else:
        asyncio.run(_synthesize_text_async(text, voice, temp_mp3_path))


def synthesize_segments(
    segments: List[Dict[str, Any]],
    output_dir: str,
    voice: Optional[str] = None,
    gender: Optional[str] = None,
    source_audio_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Synthesizes natural English speech for each segment and synchronizes its duration.

    Speaker Matching & Voice Selection:
    -----------------------------------
    - If 'voice' is explicitly provided, uses that voice directly.
    - If 'gender' is provided ('male' or 'female'), picks the matching voice.
    - Otherwise, automatically locates the original source audio (from source_audio_path,
      segment metadata, or samples/input) and calls guess_gender() once to detect pitch.
    - Maps:
        male -> 'en-US-ChristopherNeural'
        female -> 'en-US-JennyNeural'
        unspecified / default -> 'en-US-AriaNeural'
    - Prints detected gender AND the chosen voice name together in the terminal output.

    Duration Synchronization Strategy:
    ----------------------------------
    1. Small mismatches (<= 1.0s):
       No time-stretching is applied (100% natural speech rate).
       Pads trailing silence if shorter; trims end with fade-out if slightly longer.
    2. Significant mismatches (> 1.0s):
       Uses ffmpeg's pitch-preserving 'atempo' filter to time-stretch/compress the clip.
       Capped strictly within +/-15% (0.85x to 1.15x). If a segment would require more,
       a warning is logged and the factor is clamped to preserve natural speech.

    Args:
        segments: List of segment dicts containing 'start', 'end', and 'english_text'.
        output_dir: Directory where individual audio clips will be saved.
        voice: Optional specific edge-tts voice.
        gender: Optional speaker gender ('male' or 'female').
        source_audio_path: Optional path to original audio for gender detection.

    Returns:
        The updated list of segment dictionaries, each now containing 'audio_path'.
    """
    if not segments:
        return []

    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    temp_dir = os.path.join(abs_output_dir, "_temp_tts")
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Resolve source audio path if not provided directly
    if not source_audio_path:
        for seg in segments:
            for k in ("source_audio_path", "source_audio", "original_audio", "audio_path", "source"):
                cand = seg.get(k)
                if cand and isinstance(cand, str) and os.path.exists(cand):
                    # Ensure it's not a newly generated output segment file
                    if not os.path.basename(cand).startswith("segment_"):
                        source_audio_path = cand
                        break
            if source_audio_path:
                break

    # If still not found, check samples/input for available source wav files
    if not source_audio_path and os.path.exists("samples/input"):
        input_wavs = [
            os.path.join("samples/input", f)
            for f in os.listdir("samples/input")
            if f.endswith(".wav")
        ]
        if input_wavs:
            # Pick the most recently modified source wav
            input_wavs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            source_audio_path = input_wavs[0]

    # 2. Determine gender and select voice
    if voice:
        selected_voice = voice
        print(f"[synthesize] Voice explicitly specified: '{selected_voice}' (overrides gender)", flush=True)
    elif gender:
        # Manual override: skip guess_gender()/_estimate_gender_and_confidence() entirely
        g_clean = gender.strip().lower()
        if g_clean == "male":
            selected_voice = VOICE_MALE
            print(f"[synthesize] Gender: male (manual override) -> Selected voice: {selected_voice}", flush=True)
        elif g_clean == "female":
            selected_voice = VOICE_FEMALE
            print(f"[synthesize] Gender: female (manual override) -> Selected voice: {selected_voice}", flush=True)
        else:
            selected_voice = VOICE_DEFAULT
            print(f"[synthesize] Gender: {gender} (manual override) -> Selected voice: {selected_voice}", flush=True)
    else:
        # No manual override provided: fall back to auto-detection
        if source_audio_path and os.path.exists(source_audio_path):
            detected_gender, confidence, avg_f0 = _estimate_gender_and_confidence(source_audio_path)
            if detected_gender == "male":
                selected_voice = VOICE_MALE
                print(
                    f"[synthesize] Gender: male (auto-detected, confidence: {confidence:.1%}, avg F0: {avg_f0:.1f}Hz) "
                    f"-> Selected voice: {selected_voice}",
                    flush=True,
                )
            elif detected_gender == "female":
                selected_voice = VOICE_FEMALE
                print(
                    f"[synthesize] Gender: female (auto-detected, confidence: {confidence:.1%}, avg F0: {avg_f0:.1f}Hz) "
                    f"-> Selected voice: {selected_voice}",
                    flush=True,
                )
            else:
                selected_voice = VOICE_DEFAULT
                print(
                    f"[synthesize] Gender: unspecified (auto-detected ambiguous pitch: {avg_f0:.1f}Hz, confidence: {confidence:.1%}) "
                    f"-> Selected voice: {selected_voice}",
                    flush=True,
                )
        else:
            selected_voice = VOICE_DEFAULT
            print(f"[synthesize] Gender: unspecified (auto-detection fallback, no source audio) -> Selected voice: {selected_voice}", flush=True)

    total_segments = len(segments)
    print(f"[synthesize] Synthesizing {total_segments} segments...", flush=True)

    try:
        for idx, segment in enumerate(segments):
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
            target_duration = max(0.1, end - start)
            text_to_speak = (segment.get("english_text") or segment.get("text") or "").strip()

            final_wav_path = os.path.join(abs_output_dir, f"segment_{idx:04d}.wav")
            temp_mp3 = os.path.join(temp_dir, f"raw_{idx:04d}.mp3")
            temp_wav = os.path.join(temp_dir, f"raw_{idx:04d}.wav")

            # Handle empty text with pure silence
            if not text_to_speak:
                _generate_silent_wav(final_wav_path, target_duration)
                segment["audio_path"] = os.path.abspath(final_wav_path)
                print(
                    f"[synthesize] Segment {idx + 1}/{total_segments} ({start:.2f}s - {end:.2f}s): "
                    f"empty text -> generated {target_duration:.2f}s silence",
                    flush=True,
                )
                continue

            # 1. Synthesize speech via edge-tts
            try:
                _synthesize_text(text_to_speak, selected_voice, temp_mp3)
            except Exception as err:
                print(
                    f"[synthesize] Warning: TTS failed for segment {idx + 1} ('{text_to_speak[:30]}...'): {err}. "
                    f"Falling back to silence.",
                    file=sys.stderr,
                    flush=True,
                )
                _generate_silent_wav(final_wav_path, target_duration)
                segment["audio_path"] = os.path.abspath(final_wav_path)
                continue

            # 2. Convert MP3 to 24kHz mono WAV to accurately measure duration
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_mp3,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    temp_wav,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            actual_duration = _get_audio_duration(temp_wav)
            if actual_duration <= 0.05:
                _generate_silent_wav(final_wav_path, target_duration)
                segment["audio_path"] = os.path.abspath(final_wav_path)
                continue

            mismatch = abs(actual_duration - target_duration)
            raw_duration = actual_duration

            # 3. Duration Adjustment Logic (Never cut speech; only pad/trim silence and stretch within +/-15%)
            if mismatch <= 1.0:
                # Small mismatch (<= 1s): DO NOT stretch.
                if actual_duration <= target_duration:
                    # Pad trailing silence up to target_duration
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            temp_wav,
                            "-af",
                            f"apad=whole_dur={target_duration:.3f}",
                            "-vn",
                            "-acodec",
                            "pcm_s16le",
                            "-ar",
                            "24000",
                            "-ac",
                            "1",
                            final_wav_path,
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    status_note = f"padded {target_duration - actual_duration:.2f}s silence (no stretch)"
                else:
                    # Speech is slightly longer: only trim silence, NEVER cut speech!
                    trimmed_dur = _trim_trailing_silence(temp_wav)
                    if trimmed_dur <= target_duration:
                        # After trimming trailing silence it now fits; pad to target_duration
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-i",
                                temp_wav,
                                "-af",
                                f"apad=whole_dur={target_duration:.3f}",
                                "-vn",
                                "-acodec",
                                "pcm_s16le",
                                "-ar",
                                "24000",
                                "-ac",
                                "1",
                                final_wav_path,
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                        status_note = f"trimmed trailing silence to {trimmed_dur:.2f}s and padded to target"
                    else:
                        # Full audio is speech: KEEP FULL AUDIO, DO NOT CUT WORDS!
                        shutil.copyfile(temp_wav, final_wav_path)
                        overflow = trimmed_dur - target_duration
                        warn_msg = (
                            f"[synthesize] Warning: Segment {idx + 1} will run long / overlap with next segment's start "
                            f"(duration: {trimmed_dur:.2f}s exceeds target: {target_duration:.2f}s by {overflow:.2f}s). "
                            f"Keeping full audio to prevent dropping words."
                        )
                        print(warn_msg, flush=True)
                        status_note = f"kept full audio (runs long by {overflow:.2f}s, no stretch)"
            else:
                # Significant mismatch (> 1.0s): apply pitch-preserving atempo time-stretching
                speed_factor = actual_duration / target_duration
                clamped_factor = speed_factor

                # Cap stretching strictly at +/-15% (0.85x to 1.15x) for naturalness
                if speed_factor > 1.15:
                    warn_factor = (
                        f"[synthesize] Warning: Segment {idx + 1} speed-up exceeds +15% "
                        f"(target: {target_duration:.2f}s, generated: {actual_duration:.2f}s, "
                        f"factor: {speed_factor:.2f}x). Clamping to 1.15x to preserve natural speech."
                    )
                    print(warn_factor, flush=True)
                    clamped_factor = 1.15
                elif speed_factor < 0.85:
                    warn_factor = (
                        f"[synthesize] Warning: Segment {idx + 1} slow-down exceeds -15% "
                        f"(target: {target_duration:.2f}s, generated: {actual_duration:.2f}s, "
                        f"factor: {speed_factor:.2f}x). Clamping to 0.85x to preserve natural speech."
                    )
                    print(warn_factor, flush=True)
                    clamped_factor = 0.85

                stretched_wav = os.path.join(temp_dir, f"stretched_{idx:04d}.wav")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_wav,
                        "-filter:a",
                        f"atempo={clamped_factor:.4f}",
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "24000",
                        "-ac",
                        "1",
                        stretched_wav,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

                # Post-stretch alignment: only pad silence, NEVER cut speech!
                stretched_dur = _get_audio_duration(stretched_wav)
                if stretched_dur <= target_duration:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            stretched_wav,
                            "-af",
                            f"apad=whole_dur={target_duration:.3f}",
                            "-vn",
                            "-acodec",
                            "pcm_s16le",
                            "-ar",
                            "24000",
                            "-ac",
                            "1",
                            final_wav_path,
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    status_note = f"atempo stretched ({clamped_factor:.2f}x) + padded {target_duration - stretched_dur:.2f}s"
                else:
                    # Clip is longer even after stretching: trim trailing silence if possible, but NEVER cut words
                    trimmed_stretched_dur = _trim_trailing_silence(stretched_wav)
                    if trimmed_stretched_dur <= target_duration:
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-i",
                                stretched_wav,
                                "-af",
                                f"apad=whole_dur={target_duration:.3f}",
                                "-vn",
                                "-acodec",
                                "pcm_s16le",
                                "-ar",
                                "24000",
                                "-ac",
                                "1",
                                final_wav_path,
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                        status_note = f"atempo stretched ({clamped_factor:.2f}x) + trimmed trailing silence"
                    else:
                        shutil.copyfile(stretched_wav, final_wav_path)
                        overflow = trimmed_stretched_dur - target_duration
                        warn_msg = (
                            f"[synthesize] Warning: Segment {idx + 1} will run long / overlap with next segment's start "
                            f"(duration: {trimmed_stretched_dur:.2f}s exceeds target: {target_duration:.2f}s by {overflow:.2f}s). "
                            f"Keeping full audio to prevent dropping words."
                        )
                        print(warn_msg, flush=True)
                        status_note = f"atempo stretched ({clamped_factor:.2f}x) (runs long by {overflow:.2f}s)"

            final_written_duration = _get_audio_duration(final_wav_path)
            segment["audio_path"] = os.path.abspath(final_wav_path)

            print(
                f"[synthesize] Segment {idx + 1}/{total_segments} ({start:.2f}s - {end:.2f}s): "
                f"raw duration: {raw_duration:.2f}s vs. final written duration: {final_written_duration:.2f}s "
                f"(target: {target_duration:.2f}s) [{status_note}]",
                flush=True,
            )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"[synthesize] All {total_segments} segments synthesized successfully.", flush=True)
    return segments


if __name__ == "__main__":
    test_audio = os.path.join("samples", "input", "QU1Fk-XzT-A.wav")
    if os.path.exists(test_audio):
        gender_res = guess_gender(test_audio)
        print(f"guess_gender('{test_audio}') -> {gender_res}")

    sample_segments = [
        {
            "start": 0.0,
            "end": 3.5,
            "text": "Hello world, this is automated video dubbing.",
            "english_text": "Hello world, this is automated video dubbing.",
        },
    ]

    out_folder = os.path.join("samples", "output", "test_wiring")
    results = synthesize_segments(sample_segments, out_folder, source_audio_path=test_audio)
    for s in results:
        print("Segment output:", s["audio_path"])
