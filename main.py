import argparse
import os
from pathlib import Path

from src.transcriber import transcribe_m4a
from src.summarizer import summarize_text


def default_output_paths(input_path: Path, transcript_out: str | None, summary_out: str | None):
    stem = input_path.with_suffix("")
    t_out = Path(transcript_out) if transcript_out else Path(f"{stem}.transcript.txt")
    s_out = Path(summary_out) if summary_out else Path(f"{stem}.summary.txt")
    return t_out, s_out


def main():
    parser = argparse.ArgumentParser(description="Transcribe an M4A file with Whisper and summarize with Gemma locally.")
    parser.add_argument('--input', required=True, help='Path to M4A file')
    parser.add_argument('--whisper', default='base.en', help='Whisper model size (tiny.en, base.en, small.en, medium.en)')
    parser.add_argument('--echo-transcript', default=False, type=bool, help='Echoes the active transcription to the console if set to True, otherwise it displays a progress bar.')

    parser.add_argument('--backend', choices=['ollama', 'transformers'], default='ollama', help='Summarizer backend')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama base URL')
    parser.add_argument('--ollama-model', default='gemma:2b', help='Ollama model name')

    parser.add_argument('--hf-model', default='google/gemma-2-2b-it', help='HF Transformers model id for Gemma')

    parser.add_argument('--max-summary-tokens', type=int, default=-1, help='Max new tokens for the summary')
    parser.add_argument('--temperature', type=float, default=0.2, help='Decoding temperature')

    parser.add_argument('--transcript-out', default=None, help='Path to save transcript text')
    parser.add_argument('--summary-out', default=None, help='Path to save summary text')

    args = parser.parse_args()
    print(f"Parser Arguments: {args}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Transcribing: {input_path}")
    transcript = transcribe_m4a(str(input_path), model_size=args.whisper)

    t_out, s_out = default_output_paths(input_path, args.transcript_out, args.summary_out)
    t_out.write_text(transcript, encoding='utf-8')
    print(f"Transcript saved to: {t_out}")

    print(f"Summarizing transcript with {args.ollama_model}...")
    summary = summarize_text(
        transcript,
        backend=args.backend,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        hf_model_id=args.hf_model,
        max_new_tokens=args.max_summary_tokens,
        temperature=args.temperature,
    )

    s_out.write_text(summary, encoding='utf-8')
    print(f"Summary saved to: {s_out}")


if __name__ == '__main__':
    main()
