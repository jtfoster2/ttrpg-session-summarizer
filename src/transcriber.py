from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import whisper


class FFmpegNotFound(RuntimeError):
    pass


def _require_ffmpeg():
    # Whisper uses ffmpeg via subprocess; ensure it's available
    if shutil.which('ffmpeg') is None:
        raise FFmpegNotFound(
            "FFmpeg not found in PATH. Please install FFmpeg and ensure `ffmpeg` is available."
        )


def transcribe_m4a(input_path: str | os.PathLike, model_size: str = 'base.en', verbose: bool = False) -> str:
    """Transcribe an audio file using a local Whisper model.

    Args:
        input_path: Path to the input audio file (.m4a recommended).
        model_size: Whisper model to load (tiny.en, base.en, small.en, medium.en).
        verbose: If True, enables verbose logging from Whisper, otherwise False provides a progress bar.

    Returns:
        Transcript text.
    """

    _require_ffmpeg()
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    # Load whisper model locally; model files are cached under ~/.cache/whisper
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)

    print("Beginning transcription...") # TODO: Timestamp the time that the transcription begins.
    result = model.transcribe(audio=str(p), verbose=verbose, fp16=False) # fp16=False to suppress warnings on CPUs without fp16 support
    print("Transcription complete.") # TODO: Timestamp the time that the transcription finishes.

    # Whisper returns a dict with 'text' among segments
    text = (result.get('text') or '').strip()
    return text
