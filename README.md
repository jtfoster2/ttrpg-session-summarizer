Project: Local Transcribe & Summarize (Whisper + Gemma)

This repo bootstraps a simple Python CLI that:
- Uses a local Whisper model to transcribe an input .m4a audio file.
- Uses a local Gemma model to summarize the transcription.

You can run the summarization either via:
- Ollama (recommended, simplest local setup), or
- Hugging Face Transformers (local model, heavier download; CPU works but is slow).

Requirements (Windows-friendly)
1) Python 3.10+ (3.12 OK)
2) FFmpeg installed and in PATH (required by Whisper)
   - Download: https://www.gyan.dev/ffmpeg/builds/ (or via winget: `winget install Gyan.FFmpeg`)
   - Verify: `ffmpeg -version`
3) (Option A) Ollama for Gemma [recommended]
   - Install: https://ollama.com/download
   - Pull a Gemma model: `ollama pull gemma:2b` or `ollama pull gemma2:2b-instruct`
   - Verify it runs: `ollama run gemma:2b "Hello"`
4) (Option B) Transformers backend for Gemma
   - Will download a Gemma model locally on first run (several GB).
   - Works on CPU, but will be slow. If you have a CUDA GPU, install a matching PyTorch build.

Quickstart
1) Create and activate a virtual environment (optional):
   - PowerShell: `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
2) Install dependencies:
   - `pip install -r requirements.txt`
3) Transcribe and summarize using Ollama backend:
   - `python main.py --input path\to\audio.m4a --whisper tiny --backend ollama --ollama-model gemma:2b`
4) Or using Transformers backend (CPU):
   - `python main.py --input path\to\audio.m4a --whisper base --backend transformers --hf-model google/gemma-2-2b-it`

Outputs
- Transcript is saved next to the input as `yourfile.transcript.txt` (unless you pass --transcript-out)
- Summary is saved as `yourfile.summary.txt` (unless you pass --summary-out)

CLI options
- --input: Path to .m4a (other audio types supported by ffmpeg may work).
- --whisper: Whisper model size (tiny.en, base.en, small.en, medium.en,). Default: tiny.en.
- --backend: Summarizer backend: ollama | transformers. Default: ollama.
- --ollama-url: Base URL for Ollama. Default: http://localhost:11434.
- --ollama-model: Ollama model name. Default: gemma:2b.
- --hf-model: HF Transformers model id. Default: google/gemma-2-2b-it.
- --max-summary-tokens: Max new tokens for summary. Default: 300.
- --temperature: Decoding temperature. Default: 0.2.
- --transcript-out: Output path for transcript file.
- --summary-out: Output path for summary file.

Notes
- Whisper requires ffmpeg installed and discoverable in PATH. If you see errors like `ffmpeg not found`, install/verify FFmpeg.
- Large models will be slow on CPU. For faster runs, choose smaller Whisper models (tiny/base) and smaller Gemma variants.
- If you prefer faster transcription on CPU, you can swap to `faster-whisper` with minor code changes.

License
- Gemma models are subject to Google’s Gemma license. Ensure you are permitted to download/use them locally.

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