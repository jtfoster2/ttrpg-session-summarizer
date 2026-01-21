Project: Local Transcribe & Summarize (Whisper + Gemma)

This repo bootstraps a simple Python CLI that:
- Uses a local Whisper model to transcribe an input .m4a audio file.
- Uses a local Ollama model (default: gpt-oss:20b) to summarize the transcription.

Summarization is performed via Ollama only (local model served by Ollama).

Requirements (Windows-friendly)
1) Python 3.10+ (3.12 OK)
2) FFmpeg installed and in PATH (required by Whisper)
   - Download: https://www.gyan.dev/ffmpeg/builds/ (or via winget: `winget install Gyan.FFmpeg`)
   - Verify: `ffmpeg -version`
3) Ollama for LLM summarization [required]
   - Install: https://ollama.com/download
   - Pull the model: `ollama pull gpt-oss:20b`
   - Verify it runs: `ollama run gpt-oss:20b "Hello"`

Quickstart
1) Create and activate a virtual environment (optional):
   - PowerShell: `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
2) Install dependencies:
   - `pip install -r requirements.txt`
3) Transcribe and summarize using Ollama:
   - `python main.py --input path\to\audio.m4a --whisper tiny --ollama-model gpt-oss:20b`

Outputs
- Transcript is saved next to the input as `yourfile.transcript.txt` (unless you pass --transcript-out)
- Summary is saved next to the input as `yourfile.summary.txt` (unless you pass --summary-out)

CLI options
- --input: Path to .m4a (other audio types supported by ffmpeg may work).
- --whisper: Whisper model size (tiny.en, base.en, small.en, medium.en,). Default: base.en.
- --echo-transcript: Echoes the active transcription to the console if provided and set to True, otherwise or by default it displays a progress bar.

- --ollama-url: Base URL for Ollama. Default: http://localhost:11434.
- --ollama-model: Ollama model name. Default: gpt-oss:20b.

- --max-summary-tokens: Max new tokens for summary. Default: -1 (use model default).
- --temperature: Decoding temperature. Default: 0.2.

- --transcript-out: Output path for the transcript file. Only set if this should be in a different location from the input file.
- --summary-out: Output path for the summary file. Only set if this should be in a different location from the input file.

- --stream: Stream tokens from Ollama during summarization.
- --stream-reasoning: While streaming, print the reasoning trace to console if the model provides it.
- --structured-output: Request JSON structured output from the model.
- --json-schema: Path to a JSON Schema file to enforce structured output (implies --structured-output).
- --thinking-level: Thinking level preset for gpt-oss reasoning: low | medium | high (default: medium).

Notes
- You can customize the summarization prompt used by Ollama by editing the `ollama_prompt` file at the project root. If the file is missing or unreadable, a built-in default instruction will be used.
- Whisper requires ffmpeg installed and discoverable in PATH. If you see errors like `ffmpeg not found`, install/verify FFmpeg.
- Large models will be slow on CPU. For faster runs, choose smaller Whisper models (tiny/base)
- If you prefer faster transcription on CPU, you can swap to `faster-whisper` with minor code changes.

Which Whisper Model Should I Choose? (Copied from https://whisper-api.com/blog/models/)

Whisper is an automatic speech recognition (ASR) system created by OpenAI that can convert natural speech into text. This was released as an Open Source library that you can download and run on your computer. This guide will walk you through the various Whisper models, their differences, and how to select the right one for your specific use case.

Understanding Whisper Model Sizes
Whisper comes in various sizes, each representing a different tradeoff between accuracy, speed, and resource requirements. Let’s examine each of them:

Tiny Models
tiny: The smallest multilingual model (39M parameters)
tiny.en: English-only variant of tiny
Best for: Quick transcriptions where perfect accuracy isn’t critical, or when running on devices with very limited resources. If your audio is clear with minimal background noise, this model can be surprisingly effective.

Base Models
base: Small multilingual model (74M parameters)
base.en: English-only variant of base
Best for: General purpose transcription with reasonable accuracy when resources are limited. This is a great balance between speed and accuracy.

Small Models
small: Medium-sized multilingual model (244M parameters)
small.en: English-only variant of small
Best for: Daily transcription needs with good accuracy and reasonable speed. More accurate than tiny and base models but requires more resources. If you have a decent GPU or beefy CPU, this is a good choice.

Medium Models
medium: Large multilingual model (769M parameters)
medium.en: English-only variant of medium
Best for: High-quality transcriptions where accuracy is important and you have decent computing resources.

Large Models
large-v1: Original large model (1.5B parameters)
large-v2: Improved large model
large-v3: Latest large model with the best accuracy
large: Alias for the latest large model
Best for: Professional transcription where maximum accuracy is essential. If you can’t compromise on quality, this is the model to use.

Key Factors to Consider When Choosing a Model
1. Audio Quality and Complexity
The quality of your audio significantly impacts which model you should choose:

High-quality, clear audio: Smaller models like base or small might perform adequately.
Challenging audio (background noise, multiple speakers, heavy accents): Larger models like medium or large will yield better results.
2. Language Requirements
Whisper offers both multilingual and English-only models:

English-only audio: The .en variants (like tiny.en, base.en) are optimized specifically for English and generally perform better on English content while using fewer resources.
Multilingual content: The standard models without the .en suffix support 100+ languages.
3. Available Compute Resources
Your hardware constraints will significantly influence your choice:

CPU-only systems: Stick to tiny or base models for reasonable processing times.
Consumer GPUs (e.g., GTX 1060, RTX 3060): small and medium models run well.
High-end GPUs or cloud GPU instances: Can efficiently run large models.
4. Speed vs. Accuracy Tradeoffs
Consider your priorities:

Speed priority: Use tiny, base, or the turbo variants.
Accuracy priority: Use medium or large models.
Balanced approach: Consider small models or distilled variants.