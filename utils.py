"""
utils.py - Shared data models, type definitions, and utility helpers for automatedub.

Provides standard TypedDicts and dataclasses representing the shared contract between
downloader, transcriber, synthesizer, and remixer modules.
"""

from dataclasses import dataclass
import os
import sys
from typing import Any, Dict, List, Optional, TypedDict


class WordTimestamp(TypedDict, total=False):
    """Word-level timing information."""
    word: str
    start: float
    end: float
    probability: float


class SegmentDict(TypedDict, total=False):
    """
    Contract for speech segments shared across pipeline stages.

    Keys:
        start: Start timestamp in seconds (float).
        end: End timestamp in seconds (float).
        text: Original transcribed source text (str).
        english_text: Translated English text to be synthesized (str).
        audio_path: Optional path to the synthesized speech WAV clip (str).
        word_starts: Optional list of word-level timestamp dicts.
        source_audio_path: Optional path to the original extracted audio track.
    """
    start: float
    end: float
    text: str
    english_text: str
    audio_path: Optional[str]
    word_starts: Optional[List[Dict[str, Any]]]
    source_audio_path: Optional[str]


@dataclass
class Segment:
    """
    Dataclass representation of a speech segment for structured manipulation.
    """
    start: float
    end: float
    text: str = ""
    english_text: str = ""
    audio_path: Optional[str] = None
    word_starts: Optional[List[Dict[str, Any]]] = None
    source_audio_path: Optional[str] = None

    @property
    def duration(self) -> float:
        """Target duration of the segment in seconds."""
        return max(0.0, self.end - self.start)

    def to_dict(self) -> SegmentDict:
        """Converts segment dataclass instance to a segment dictionary."""
        d: SegmentDict = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "english_text": self.english_text or self.text,
        }
        if self.audio_path is not None:
            d["audio_path"] = self.audio_path
        if self.word_starts is not None:
            d["word_starts"] = self.word_starts
        if self.source_audio_path is not None:
            d["source_audio_path"] = self.source_audio_path
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Segment":
        """Creates a Segment instance from a dictionary."""
        return cls(
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
            text=str(d.get("text") or ""),
            english_text=str(d.get("english_text") or d.get("text") or ""),
            audio_path=d.get("audio_path"),
            word_starts=d.get("word_starts"),
            source_audio_path=d.get("source_audio_path"),
        )


class DownloadResult(TypedDict):
    """Contract for downloader.download_video return value."""
    video_path: str
    audio_path: str
    duration: float


class RemixStats(TypedDict, total=False):
    """Contract for stitching / remixing statistics."""
    speech_duration: float
    silence_duration: float
    total_duration: float
    num_segments: int
    num_overlaps: int


class AutomatedubError(Exception):
    """Base exception for all automatedub pipeline errors."""
    pass


class TranscribeError(AutomatedubError):
    """Raised when audio transcription or translation fails."""
    pass


class SynthesizeError(AutomatedubError):
    """Raised when speech synthesis fails."""
    pass


def format_duration(seconds: float) -> str:
    """Format duration into readable HH:MM:SS or MM:SS."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:05.2f}"
    return f"{mins:02d}:{secs:05.2f}"


def print_stage_header(stage_num: int, title: str) -> None:
    """Prints a standardized stage banner to stdout."""
    banner = f"=== STAGE {stage_num}: {title.upper()} ==="
    print(f"\n{banner}", flush=True)
