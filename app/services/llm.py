"""Provider-agnostic LLM wrapper.

- gemini (default): free tier via google-genai SDK
- groq: free tier via OpenAI-compatible API
- offline fallback: when no API key is configured every call returns None and
  the callers use extractive heuristics so the app stays fully demoable.
"""
from __future__ import annotations

import json
import re

import httpx

from ..config import settings

GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"


class LLMUnavailable(Exception):
    pass


def llm_available() -> bool:
    if settings.llm_provider == "groq":
        return bool(settings.groq_api_key)
    return bool(settings.gemini_api_key)


def _gemini_call(prompt: str, system: str, json_mode: bool, temperature: float) -> str | None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    cfg = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
    )
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=cfg)
    return resp.text


def _groq_call(prompt: str, system: str, json_mode: bool, temperature: float) -> str | None:
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={
            "model": GROQ_MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **({"response_format": {"type": "json_object"}} if json_mode else {}),
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate(prompt: str, system: str = "", json_mode: bool = False, temperature: float = 0.4) -> str | None:
    """Call the configured provider. Returns None when unavailable/offline."""
    if not llm_available():
        return None
    try:
        if settings.llm_provider == "groq":
            return _groq_call(prompt, system, json_mode, temperature)
        return _gemini_call(prompt, system, json_mode, temperature)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        print(f"[llm] call failed: {exc}")
        return None


def generate_json(prompt: str, system: str = "", temperature: float = 0.4) -> dict | list | None:
    raw = generate(prompt, system=system, json_mode=True, temperature=temperature)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


# ------------------------- extractive fallbacks -------------------------


def extractive_summary(text: str, max_sentences: int = 6) -> str:
    from .rag import split_sentences

    sentences = split_sentences(text)
    if not sentences:
        return ""
    # rank sentences by word frequency (classic extractive heuristic)
    freq: dict[str, int] = {}
    for s in sentences:
        for w in re.findall(r"[a-zA-Z]{4,}", s.lower()):
            freq[w] = freq.get(w, 0) + 1
    def score(s: str) -> int:
        return sum(freq.get(w, 0) for w in re.findall(r"[a-zA-Z]{4,}", s.lower()))
    ranked = sorted(sentences, key=score, reverse=True)[:max_sentences]
    ordered = [s for s in sentences if s in ranked]
    return " ".join(ordered)


def top_keywords(text: str, k: int = 6) -> list[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "your", "have",
        "will", "been", "which", "their", "about", "would", "these", "other", "than",
        "more", "when", "what", "where", "such", "also", "each", "they", "them",
        "there", "then", "were", "are", "was", "can", "not", "but", "has", "had",
        "its", "it's", "you", "any", "all", "one", "two", "how", "why", "who",
    }
    freq: dict[str, int] = {}
    for w in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text.lower()):
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [w.capitalize() for w, _ in ranked[:k]]
