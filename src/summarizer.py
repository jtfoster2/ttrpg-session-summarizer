from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import requests

# Lazy import for transformers to keep ollama-only setups light

# "You are a helpful assistant. Summarize the following transcript clearly and concisely. "
#  "Highlight key points, decisions, and action items. Keep it under ~10 sentences.\n\n"

def _format_prompt(text: str) -> str:
    instruction = (
        "Summarize the following transcript from a tabletop roleplaying session in a thorough, narrative style for a recap. Cover as many important details as possible. Highlight key events"
        "Highlight key events, where the party ended the session at, and mention some notable quotes from the transcript. Also provide a list of all characters who speak in the transcription\n\n"
    )
    return f"{instruction}Transcript:\n{text}\n\nSummary:"


@dataclass
class SummarizeConfig:
    backend: str = 'ollama'  # 'ollama' or 'transformers'
    ollama_url: str = 'http://localhost:11434'
    ollama_model: str = 'gemma:2b'
    hf_model_id: str = 'google/gemma-2-2b-it'
    max_new_tokens: int = -1
    temperature: float = 0.2


class SummarizationError(RuntimeError):
    pass


def _summarize_with_ollama(prompt: str, url: str, model: str, max_tokens: int, temperature: float) -> str:
    endpoint = f"{url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 65366
        },
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=600)
    except requests.RequestException as e:
        raise SummarizationError(f"Failed to connect to Ollama at {endpoint}: {e}")

    if resp.status_code != 200:
        raise SummarizationError(f"Ollama error {resp.status_code}: {resp.text}")

    data = resp.json()
    return (data.get('response') or '').strip()


def _summarize_with_transformers(prompt: str, model_id: str, max_tokens: int, temperature: float) -> str:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as e:
        raise SummarizationError(
            f"Transformers backend is not available: {e}. Install dependencies in requirements.txt"
        )

    device = 'cuda' if hasattr(torch, 'cuda') and torch.cuda.is_available() else 'cpu'

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    do_sample = temperature > 0.0
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        top_p=0.9 if do_sample else None,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return only the assistant continuation after 'Summary:' if present
    if 'Summary:' in text:
        text = text.split('Summary:', 1)[1]
    return text.strip()


def summarize_text(
    transcript: str,
    backend: str = 'ollama',
    ollama_url: str = 'http://localhost:11434',
    ollama_model: str = 'gemma:2b',
    hf_model_id: str = 'google/gemma-2-2b-it',
    max_new_tokens: int = -1,
    temperature: float = 0.2,
) -> str:
    """Summarize a transcript using a local Gemma model via Ollama or Transformers.
    """
    if not transcript or not transcript.strip():
        return "(Empty transcript)"

    prompt = _format_prompt(transcript)

    if backend == 'ollama':
        return _summarize_with_ollama(prompt, ollama_url, ollama_model, max_new_tokens, temperature)
    elif backend == 'transformers':
        return _summarize_with_transformers(prompt, hf_model_id, max_new_tokens, temperature)
    else:
        raise SummarizationError(f"Unknown backend: {backend}")
