"""
main.py - CLI entry point for the automatedub video dubbing pipeline.

Orchestrates the 4 end-to-end pipeline stages:
1. Downloader: Downloads video and extracts 16kHz mono audio.
2. Transcriber: Speech-to-text and translation into timed English segments via Whisper.
3. Synthesizer: Text-to-speech with gender matching and duration synchronization via edge-tts.
4. Remixer: Precise timeline audio stitching and video remuxing via ffmpeg.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import downloader
from downloader import DownloadError
import remixer
from remixer import RemixError
import synthesizer
import transcriber
import utils
from utils import print_stage_header


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for the automatedub CLI."""
    parser = argparse.ArgumentParser(
        description="Automatedub - End-to-end automated AI video dubbing pipeline."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="YouTube video URL to download and dub into English.",
    )
    parser.add_argument(
        "--url",
        dest="flag_url",
        default=None,
        help="YouTube video URL (alternative to positional argument).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="samples/output/",
        help="Directory where intermediate assets and the final video will be saved (default: 'samples/output/').",
    )
    parser.add_argument(
        "--voice-gender",
        choices=["male", "female"],
        default=None,
        help="Optional override for dubbed voice gender ('male' or 'female'). If omitted, gender is auto-detected from original audio.",
    )
    return parser.parse_args(argv)


def run_pipeline(
    url: str,
    output_dir: str = "samples/output/",
    voice_gender: Optional[str] = None,
) -> str:
    """
    Executes the full automated dubbing pipeline:
    download -> transcribe -> synthesize -> remix.

    Args:
        url: YouTube video URL.
        output_dir: Directory where outputs and assets are saved.
        voice_gender: Optional manual override for voice gender ('male' or 'female').
                      If None, synthesizer auto-detects gender from original audio.

    Returns:
        The absolute path to the final dubbed video file.
    """
    total_start_time = time.time()
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # STAGE 1: DOWNLOAD
    # -------------------------------------------------------------------------
    print_stage_header(1, "DOWNLOAD")
    stage_start = time.time()
    try:
        download_res = downloader.download_video(url, abs_output_dir)
        video_path = download_res["video_path"]
        audio_path = download_res["audio_path"]
        video_duration = download_res.get("duration", 0.0)
    except DownloadError as err:
        print(f"[main] Download failed: {err}", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as err:
        print(f"[main] Unexpected error during download: {err}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(
        f"[main] Stage 1 complete in {time.time() - stage_start:.2f}s: "
        f"video='{video_path}', audio='{audio_path}' ({video_duration:.2f}s)",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STAGE 2: TRANSCRIBE
    # -------------------------------------------------------------------------
    print_stage_header(2, "TRANSCRIBE & TRANSLATE")
    stage_start = time.time()
    try:
        segments = transcriber.transcribe(audio_path)
    except (FileNotFoundError, ValueError) as err:
        print(f"[main] Transcription failed: {err}", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as err:
        print(f"[main] Unexpected error during transcription: {err}", file=sys.stderr, flush=True)
        sys.exit(1)

    if not segments:
        print("[main] Warning: No speech segments were detected in the audio.", flush=True)

    print(
        f"[main] Stage 2 complete in {time.time() - stage_start:.2f}s: "
        f"transcribed {len(segments)} segments.",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STAGE 3: SYNTHESIZE
    # -------------------------------------------------------------------------
    print_stage_header(3, "SYNTHESIZE SPEECH")
    stage_start = time.time()
    try:
        synth_kwargs: Dict[str, Any] = {
            "segments": segments,
            "output_dir": abs_output_dir,
            "source_audio_path": audio_path,
        }
        # When not provided by user, do NOT pass gender so auto-detection is used
        if voice_gender is not None:
            synth_kwargs["gender"] = voice_gender

        segments = synthesizer.synthesize_segments(**synth_kwargs)
    except Exception as err:
        print(f"[main] Speech synthesis failed: {err}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(
        f"[main] Stage 3 complete in {time.time() - stage_start:.2f}s: "
        f"synthesized {len(segments)} audio segments.",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STAGE 4: REMIX
    # -------------------------------------------------------------------------
    print_stage_header(4, "REMIX & MUX")
    stage_start = time.time()

    # Determine output video file name in output_dir
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    final_video_name = f"{video_basename}_dubbed.mp4"
    final_video_path = os.path.join(abs_output_dir, final_video_name)

    try:
        remixed_path = remixer.remix(video_path, segments, final_video_path)
    except RemixError as err:
        print(f"[main] Remixing failed: {err}", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as err:
        print(f"[main] Unexpected error during remixing: {err}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(
        f"[main] Stage 4 complete in {time.time() - stage_start:.2f}s: "
        f"output='{remixed_path}'",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # SUMMARY & ELAPSED TIME
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - total_start_time
    mins = int(total_elapsed // 60)
    secs = total_elapsed % 60
    print("\n" + "=" * 50, flush=True)
    print("[main] Pipeline finished successfully!", flush=True)
    print(f"[main] Total elapsed time: {total_elapsed:.2f}s ({mins}m {secs:.2f}s)", flush=True)
    print(f"[main] Final dubbed video: {remixed_path}", flush=True)
    print("=" * 50 + "\n", flush=True)

    return remixed_path


def main() -> None:
    args = parse_args()

    url = (args.url or args.flag_url or "").strip()
    if not url:
        try:
            url = input("Enter YouTube URL: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[main] Operation cancelled by user.", file=sys.stderr)
            sys.exit(1)

    if not url:
        print("[main] Error: A valid YouTube URL must be provided.", file=sys.stderr)
        sys.exit(1)

    run_pipeline(
        url=url,
        output_dir=args.output_dir,
        voice_gender=args.voice_gender,
    )


if __name__ == "__main__":
    main()
