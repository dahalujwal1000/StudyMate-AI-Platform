"""Provider-agnostic LLM wrapper with automatic fallback.

- gemini (default): free tier via google-genai SDK
- mistral: free tier via OpenAI-compatible API (https://api.mistral.ai)
- groq: free tier via OpenAI-compatible API
- fallback chain: the configured provider is tried first; if it errors, any
  other configured provider is tried next (e.g. gemini -> mistral -> groq).
- offline: when NO provider has a key, every call returns None and callers use
  extractive heuristics so the app stays fully demoable.
"""
from __future__ import annotations

import json
import re

import httpx

from ..config import settings

GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_MODEL_FALLBACKS = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MODEL_FALLBACKS = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "allam-2-7b"]
MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_MODEL_FALLBACKS = ["open-mistral-nemo", "mistral-medium-latest"]

ALL_PROVIDERS = ("gemini", "mistral", "groq")


def _provider_keys() -> dict[str, str]:
    return {
        "gemini": (settings.gemini_api_key or "").strip(),
        "mistral": (settings.mistral_api_key or "").strip(),
        "groq": (settings.groq_api_key or "").strip(),
    }


def llm_available() -> bool:
    return any(_provider_keys().values())


def _provider_order() -> list[str]:
    """Configured provider first; then every other provider that has a key."""
    keys = _provider_keys()
    primary = settings.llm_provider
    order: list[str] = []
    if primary in keys and keys[primary]:
        order.append(primary)
    for p in ALL_PROVIDERS:
        if p not in order and keys.get(p):
            order.append(p)
    return order


def _gemini_models() -> list[str]:
    """Models to try in order (settings override first)."""
    primary = (settings.gemini_model or GEMINI_MODEL).strip()
    models = [primary]
    for m in [GEMINI_MODEL, *GEMINI_MODEL_FALLBACKS]:
        if m not in models:
            models.append(m)
    return models


def _mistral_models() -> list[str]:
    primary = (settings.mistral_model or MISTRAL_MODEL).strip()
    models = [primary]
    for m in [MISTRAL_MODEL, *MISTRAL_MODEL_FALLBACKS]:
        if m not in models:
            models.append(m)
    return models


def _groq_models() -> list[str]:
    primary = GROQ_MODEL
    models = [primary]
    for m in [GROQ_MODEL, *GROQ_MODEL_FALLBACKS]:
        if m not in models:
            models.append(m)
    return models


def _gemini_call(prompt: str, system: str, json_mode: bool, temperature: float) -> str | None:
    from google import genai
    from google.genai import types

    cfg = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
    )
    errors: list[str] = []
    for model in _gemini_models():
        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            if resp and resp.text:
                return resp.text
            break
        except Exception as exc:  # try the next available model
            errors.append(f"{model}: {exc}")
            if "NOT_FOUND" not in str(exc):
                break
    raise RuntimeError(" | ".join(errors) if errors else "gemini returned no text")


def _mistral_call(prompt: str, system: str, json_mode: bool, temperature: float) -> str | None:
    errors: list[str] = []
    for model in _mistral_models():
        try:
            resp = httpx.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.mistral_api_key}"},
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    **({"response_format": {"type": "json_object"}} if json_mode else {}),
                },
                timeout=60,
            )
            if resp.status_code == 404:
                errors.append(f"{model}: model not found")
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            errors.append(f"{model}: {exc.response.status_code} {exc.response.text[:120]}")
            if exc.response.status_code == 404:
                continue
            raise
        except Exception as exc:
            errors.append(f"{model}: {exc}")
            raise
    raise RuntimeError(" | ".join(errors) if errors else "mistral returned no text")


def _groq_call(prompt: str, system: str, json_mode: bool, temperature: float) -> str | None:
    errors: list[str] = []
    for model in _groq_models():
        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    **({"response_format": {"type": "json_object"}} if json_mode else {}),
                },
                timeout=90,
            )
            if resp.status_code == 404:
                errors.append(f"{model}: model not found")
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            errors.append(f"{model}: {exc.response.status_code} {exc.response.text[:160]}")
            if exc.response.status_code == 404:
                continue
            raise
        except Exception as exc:
            errors.append(f"{model}: {exc}")
            raise
    raise RuntimeError(" | ".join(errors) if errors else "groq returned no text")


def _openai_chat(
    url: str,
    api_key: str,
    models: list[str],
    messages: list[dict],
    json_mode: bool,
    temperature: float,
) -> str | None:
    """OpenAI-compatible multi-turn call that tries each model in order."""
    errors: list[str] = []
    for model in models:
        try:
            resp = httpx.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": messages,
                    **({"response_format": {"type": "json_object"}} if json_mode else {}),
                },
                timeout=90,
            )
            if resp.status_code == 404:
                errors.append(f"{model}: model not found")
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            errors.append(f"{model}: {exc.response.status_code} {exc.response.text[:160]}")
            if exc.response.status_code == 404:
                continue
            raise
        except Exception as exc:
            errors.append(f"{model}: {exc}")
            raise
    raise RuntimeError(" | ".join(errors) if errors else "provider returned no text")


def generate_chat(
    prompt: str,
    system: str = "",
    history: list[tuple[str, str]] | None = None,
    json_mode: bool = False,
    temperature: float = 0.3,
) -> str | None:
    """Multi-turn aware generation — the full conversation is sent to the model.

    history is [(role, content), ...] with role in {\"user\", \"assistant\"} so the
    model actually follows prior turns instead of answering each question in a vacuum.
    """
    history = history or []
    order = _provider_order()
    if not order:
        return None

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    for provider in order:
        try:
            if provider == "groq":
                return _openai_chat(
                    "https://api.groq.com/openai/v1/chat/completions",
                    settings.groq_api_key,
                    _groq_models(),
                    messages,
                    json_mode,
                    temperature,
                )
            if provider == "mistral":
                return _openai_chat(
                    "https://api.mistral.ai/v1/chat/completions",
                    settings.mistral_api_key,
                    _mistral_models(),
                    messages,
                    json_mode,
                    temperature,
                )
            if provider == "gemini":
                _mixed = "".join(f"\n\n{role.capitalize()}: {content}" for role, content in history)
                return _gemini_call(prompt + _mixed, system, json_mode, temperature)
        except Exception as exc:  # noqa: BLE001 - try the next provider
            print(f"[llm] {provider} failed: {exc}")
    return None


def generate(prompt: str, system: str = "", json_mode: bool = False, temperature: float = 0.4) -> str | None:
    """Call the first working provider. Returns None when fully offline."""
    order = _provider_order()
    if not order:
        return None
    for provider in order:
        try:
            if provider == "gemini":
                return _gemini_call(prompt, system, json_mode, temperature)
            if provider == "mistral":
                return _mistral_call(prompt, system, json_mode, temperature)
            if provider == "groq":
                return _groq_call(prompt, system, json_mode, temperature)
        except Exception as exc:  # noqa: BLE001 - try the next provider
            print(f"[llm] {provider} failed: {exc}")
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
