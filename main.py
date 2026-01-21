import argparse
import os
import json
from pathlib import Path

from src.transcriber import transcribe_m4a
from src.summarizer import summarize_text


def default_output_paths(input_path: Path, transcript_out: str | None, summary_out: str | None):
    stem = input_path.with_suffix("")
    t_out = Path(transcript_out) if transcript_out else Path(f"{stem}.transcript.txt")
    s_out = Path(summary_out) if summary_out else Path(f"{stem}.summary.txt")
    return t_out, s_out


def main():
    parser = argparse.ArgumentParser(description="Transcribe an M4A file with Whisper and summarize with a Ollama LLM.")
    parser.add_argument('--input', required=True, help='Path to M4A file to transcribe and summarize. Path from the current working directory.')
    parser.add_argument('--whisper', default='base.en', help='Whisper model size (tiny.en, base.en, small.en, medium.en). Defaults to base.en')
    parser.add_argument('--echo-transcript', default=False, type=bool, help='Echoes the active transcription to the console if provided and set to True, otherwise or by default it displays a progress bar.')

    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama\'s URL for LLM calls. Defaults to localhost:11434')
    parser.add_argument('--ollama-model', default='gpt-oss:20b', help='Name of the Ollama model to use for summarization')

    parser.add_argument('--max-summary-tokens', type=int, default=1500, help='Max new tokens for the summary')
    parser.add_argument('--temperature', type=float, default=1.0, help='Decoding temperature')

    parser.add_argument('--transcript-out', default=None, help='Path to save transcript text. Only set if this should be in a different location from the input file.')
    parser.add_argument('--summary-out', default=None, help='Path to save summary text. Only set if this should be in a different location from the input file.')

    # Streaming and gpt-oss thinking / structured output options
    parser.add_argument('--stream', action='store_true', help='Stream tokens from Ollama during summarization')
    parser.add_argument('--stream-reasoning', action='store_true', help='While streaming, print the reasoning trace to console if the model provides it')
    parser.add_argument('--structured-output', action='store_true', help='Request JSON structured output from the model')
    parser.add_argument('--json-schema', default=None, help='Path to a JSON Schema file to enforce structured output (implies --structured-output)')
    parser.add_argument('--thinking-level', choices=['low', 'medium', 'high'], default='medium', help='Thinking level preset for gpt-oss reasoning (low, medium, or high). Defaults to medium.')

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

    # Load JSON schema and choose thinking preset
    json_schema = None
    if args.json_schema:
        try:
            json_schema = json.loads(Path(args.json_schema).read_text(encoding='utf-8'))
        except Exception as e:
            print(f"Failed to read JSON schema from {args.json_schema}: {e}")
            json_schema = None

    summary = summarize_text(
        transcript,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        max_new_tokens=args.max_summary_tokens,
        temperature=args.temperature,
        stream=args.stream,
        stream_reasoning=args.stream_reasoning,
        structured_output=args.structured_output or bool(json_schema),
        json_schema=json_schema,
        thinking=args.thinking_level,
    )

    s_out.write_text(summary, encoding='utf-8')
    print(f"Summary saved to: {s_out}")


if __name__ == '__main__':
    main()
