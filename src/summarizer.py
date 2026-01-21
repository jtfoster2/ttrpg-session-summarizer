from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import requests
from pathlib import Path



def _format_prompt(text: str) -> str:
    """Build the prompt for Ollama by reading the base instruction from 'ollama_prompt'.

    If the file is missing or unreadable, fall back to a sensible default instruction.
    """
    instruction = None
    try:
        project_root = Path(__file__).resolve().parent.parent
        prompt_path = project_root / 'ollama_prompt'
        if prompt_path.exists():
            raw = prompt_path.read_text(encoding='utf-8').strip()
            # If the file content is wrapped in quotes, strip them
            if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                raw = raw[1:-1]
            instruction = raw
    except Exception:
        instruction = None

    if not instruction:
        instruction = (
            "Summarize the following transcript from a session of a Pathfinder: Kingmaker tabletop roleplaying game. "
            "Provide a full recap of what occurred during the session, in a style similar to the recap of a previous "
            "session early on in this transcript. Also mention where the party ended the session at, and mention some "
            "notable quotes from the transcript."
        )

    print(f"Loaded Ollama Prompt: {instruction}")
    return f"{instruction}Transcript:\n{text}\n\nSummary:"

def _summarize_with_ollama(
    prompt: str,
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    stream: bool = False,
    stream_reasoning: bool = False,
    structured_output: bool = False,
    json_schema: Optional[dict] = None,
    thinking: Optional[dict] = None,
) -> str:

    endpoint = f"{url.rstrip('/')}/api/generate"
    # Define the payload with the required fields
    # Ollama API Reference: https://docs.ollama.com/api/generate
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": bool(stream),
        "think": thinking,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    # Structured output via format, either simple "json" or a full JSON schema
    if structured_output:
        if json_schema:
            payload["format"] = json_schema
        else:
            payload["format"] = "json"

    try:
        resp = requests.post(endpoint, json=payload, stream=stream)

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to connect to Ollama at {endpoint}: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

    # Handle streaming responses
    if stream:
        full_response = []
        reasoning_buf = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                # Print raw line for debugging if it isn't JSON
                continue

            # Some Ollama models may stream "reasoning" separately
            reason = chunk.get("reasoning") or chunk.get("thinking")
            if reason:
                reasoning_buf.append(reason)
                if stream_reasoning:
                    print(reason, end="", flush=True)

            text = chunk.get("response")
            if text:
                full_response.append(text)

            if chunk.get("done") is True:
                break
        return ("".join(full_response)).strip()

    # Non-streaming path
    data = resp.json()
    return (data.get('response') or '').strip()

def summarize_text(
    transcript: str,
    ollama_url: str = 'http://localhost:11434',
    ollama_model: str = 'gpt-oss:20b',
    max_new_tokens: int = 1500,
    temperature: float = 1.0,
    stream: bool = False,
    stream_reasoning: bool = False,
    structured_output: bool = False,
    json_schema: Optional[dict] = None,
    thinking: Optional[dict] = None,
) -> str:
    """Summarize a transcript using a local model via Ollama only.

    Parameters:
    - stream: stream tokens from Ollama
    - stream_reasoning: while streaming, echo any reasoning trace to console
    - structured_output: if True, request JSON output; if json_schema provided, send full schema
    - json_schema: optional JSON Schema dict for structured output
    - thinking: optional dict of thinking options for gpt-oss (e.g., {"type": "brief", "budget_tokens": 2048})
    """
    if not transcript or not transcript.strip():
        return "(Empty transcript)"

    prompt = _format_prompt(transcript)

    return _summarize_with_ollama(
        prompt,
        ollama_url,
        ollama_model,
        max_new_tokens,
        temperature,
        stream=stream,
        stream_reasoning=stream_reasoning,
        structured_output=structured_output,
        json_schema=json_schema,
        thinking=thinking,
    )