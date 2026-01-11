from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, List

import whisper


class FFmpegNotFound(RuntimeError):
    pass


def _require_ffmpeg():
    # Whisper uses ffmpeg via subprocess; ensure it's available
    if shutil.which('ffmpeg') is None:
        raise FFmpegNotFound(
            "FFmpeg not found in PATH. Please install FFmpeg and ensure `ffmpeg` is available."
        )


def _load_pyannote_pipeline(local_model_dir: str):
    """Lazily import and load a local pyannote.audio diarization pipeline.

    Args:
        local_model_dir: Path to a locally available pyannote pipeline directory (e.g., community-1).

    Returns:
        A loaded pyannote.audio Pipeline instance.
    """
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise RuntimeError(
            "pyannote.audio is not installed. Please `pip install pyannote.audio` or add it to requirements.txt"
        ) from e

    local_path = Path(local_model_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Diarization model directory not found: {local_path}")

    # Load pipeline from local directory to avoid any external tokens
    try:
        pipeline = Pipeline.from_pretrained(str(local_path))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load pyannote pipeline from '{local_path}'. Ensure it contains a valid self-hosted pipeline (e.g., community-1)."
        ) from e
    return pipeline


def _assign_speakers_to_whisper_segments(diarization, whisper_segments):
    """Assign a speaker label to each whisper segment based on max temporal overlap.

    diarization: pyannote.core.Annotation
    whisper_segments: list of dicts with 'start','end','text'

    Returns list[str] where each item is formatted line with optional speaker prefix.
    """
    # Build list of diarization turns: (start, end, label)
    turns = []
    try:
        # pyannote Annotation API
        for segment, _, label in diarization.itertracks(yield_label=True):
            turns.append((float(segment.start), float(segment.end), str(label)))
    except AttributeError:
        # Fallback: newer API might expose 'segments' or be iterable of (segment, track, label)
        for item in diarization:
            # Try to unpack common patterns
            try:
                segment, label = item[0], item[-1]
                turns.append((float(segment.start), float(segment.end), str(label)))
            except Exception:
                pass

    def overlap(a_start, a_end, b_start, b_end):
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    lines = []
    for seg in whisper_segments:
        ws, we, text = float(seg.get('start', 0.0)), float(seg.get('end', 0.0)), (seg.get('text') or '').strip()
        if not text:
            continue
        # Choose speaker with maximal overlap
        best_label, best_olap = None, 0.0
        for ds, de, label in turns:
            ol = overlap(ws, we, ds, de)
            if ol > best_olap:
                best_olap, best_label = ol, label
        if best_label is None:
            line = text
        else:
            # Normalize label to a compact form like SPK01
            norm = str(best_label)
            if not norm.upper().startswith("SPK"):
                # try to map labels like 'SPEAKER_00'
                digits = ''.join(ch for ch in norm if ch.isdigit())
                if digits:
                    norm = f"SPK{int(digits):02d}"
                else:
                    norm = f"SPK{abs(hash(norm)) % 100:02d}"
            line = f"[{norm}] {text}"
        lines.append(line)
    return lines

def _apply_naive_diarization(
    whisper_segments: List[dict],
    n_speakers: int = 5,
    gap_threshold: float = 0.8,
) -> Optional[str]:
    """Tokenless diarization heuristic based on segment gaps.

    - Assumes a small number of speakers.
    - Alternates/cycles speaker id when a gap between segments exceeds gap_threshold.
    - Keeps current speaker within continuous speech blocks.
    """
    if not whisper_segments:
        return None

    # Sort by start time to be safe
    segs = sorted(
        (
            {
                'start': float(s.get('start', 0.0) or 0.0),
                'end': float(s.get('end', 0.0) or 0.0),
                'text': (s.get('text') or '').strip(),
            }
            for s in whisper_segments
        ),
        key=lambda x: x['start']
    )

    lines: List[str] = []
    current_speaker = 0
    last_end = None

    for s in segs:
        txt = s['text']
        if not txt:
            continue
        if last_end is not None:
            gap = max(0.0, s['start'] - last_end)
            if gap > gap_threshold:
                current_speaker = (current_speaker + 1) % max(1, n_speakers)
        spk_label = f"SPEAKER_{current_speaker:02d}"
        lines.append(f"{spk_label}: {txt}")
        last_end = s['end'] if s['end'] >= s['start'] else s['start']

    return "\n".join(lines) if lines else None

def transcribe_m4a(
    input_path: str | os.PathLike,
    model_size: str = 'base.en',
    verbose: bool = False,
    diarize: bool = False,
    diarization_model_path: Optional[str] = None,
    num_speakers: Optional[int] = None,
    naive_gap_threshold: Optional[float] = 0.8,
) -> str:
    """Transcribe an audio file using a local Whisper model.

    Args:
        input_path: Path to the input audio file (.m4a recommended).
        model_size: Whisper model to load (tiny.en, base.en, small.en, medium.en).
        verbose: If True, enables verbose logging from Whisper, otherwise False provides a progress bar.
        diarize: When True, apply speaker diarization and prefix lines with speaker labels.
        diarization_model_path: Path to a local pyannote 4.0 self-hosted pipeline directory (e.g., community-1).
        num_speakers: Optional fixed number of speakers to guide diarization.
        naive_gap_threshold: Gap seconds threshold for naive diarization (if no diarization_model_path provided).

    Returns:
        Transcript text. If diarize=True, returns multi-line speaker-attributed transcript.
    """

    _require_ffmpeg()
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    # Load whisper model locally; model files are cached under ~/.cache/whisper
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)

    print("Beginning transcription...")  # TODO: Timestamp the time that the transcription begins.
    result = model.transcribe(audio=str(p), verbose=verbose, fp16=False)  # fp16=False to suppress warnings on CPUs without fp16 support
    print("Transcription complete.")  # TODO: Timestamp the time that the transcription finishes.

    # If diarization not requested, return plain text
    if not diarize:
        text = (result.get('text') or '').strip()
        return text

    # Diarization requested: load pyannote pipeline and infer speakers
    if not diarization_model_path:
        print("Diarization requested but no diarization_model_path provided. Provide path to self-hosted 'community-1' pipeline directory. Defaulting to naive diarization based on segment gaps.")
        return _apply_naive_diarization(result.get('segments'), n_speakers=num_speakers, gap_threshold=naive_gap_threshold)

    print(f"Loading diarization pipeline from: {diarization_model_path}")
    pipeline = _load_pyannote_pipeline(diarization_model_path)

    print("Running pyannote community-1 speaker diarization...")
    # pyannote pipeline typically accepts path and optional num_speakers
    try:
        if num_speakers is not None and num_speakers > 0:
            diarization = pipeline(str(p), num_speakers=int(num_speakers))
        else:
            diarization = pipeline(str(p))
    except TypeError:
        # Some pipeline signatures use a dict with 'audio' key
        if num_speakers is not None and num_speakers > 0:
            diarization = pipeline({"audio": str(p)}, num_speakers=int(num_speakers))
        else:
            diarization = pipeline({"audio": str(p)})

    segments = result.get('segments') or []
    lines = _assign_speakers_to_whisper_segments(diarization, segments)
    # Join lines with newlines to create readable speaker-attributed transcript
    return "\n".join(lines)
