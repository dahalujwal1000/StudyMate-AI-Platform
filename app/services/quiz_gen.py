"""Generate MCQs and flashcards from document content.

Uses the LLM when configured; falls back to extractive heuristics otherwise.
"""
from __future__ import annotations

import random
import re

from . import llm
from .rag import split_sentences, tokenize

SYSTEM_QUIZ = (
    "You are an expert academic tutor creating exam-quality material. "
    "Base every item strictly on the provided study material. Respond with valid JSON only."
)


def _build_context(chunks: list[str], max_chars: int = 9000) -> str:
    out: list[str] = []
    size = 0
    for c in chunks:
        out.append(c)
        size += len(c)
        if size >= max_chars:
            break
    return "\n\n---\n\n".join(out)[:max_chars]


def generate_mcqs(title: str, chunks: list[str], n: int = 5) -> list[dict]:
    context = _build_context(chunks)
    prompt = (
        f"Study material from '{title}':\n\n{context}\n\n"
        f"Create {n} multiple-choice questions that test conceptual understanding.\n"
        'Respond as JSON: {"questions": [{"question": str, "options": [str, str, str, str], '
        '"answer_idx": 0-3, "explanation": str}]}'
    )
    data = llm.generate_json(prompt, system=SYSTEM_QUIZ, temperature=0.6)
    if data and isinstance(data, dict) and data.get("questions"):
        cleaned = []
        for q in data["questions"][:n]:
            if not q.get("question") or len(q.get("options") or []) < 2:
                continue
            cleaned.append({
                "question": str(q["question"]).strip(),
                "options": [str(o) for o in q["options"]],
                "answer_idx": int(q.get("answer_idx", 0)),
                "explanation": str(q.get("explanation", "")).strip(),
            })
        if cleaned:
            return cleaned
    return _heuristic_mcqs(title, chunks, n)


def generate_flashcards(title: str, chunks: list[str], n: int = 8) -> list[dict]:
    context = _build_context(chunks)
    prompt = (
        f"Study material from '{title}':\n\n{context}\n\n"
        f"Create {n} flashcards for active recall.\n"
        'Respond as JSON: {"cards": [{"front": str, "back": str}]}'
    )
    data = llm.generate_json(prompt, system=SYSTEM_QUIZ, temperature=0.5)
    if data and isinstance(data, dict) and data.get("cards"):
        return [
            {"front": str(c["front"]).strip(), "back": str(c.get("back", "")).strip()}
            for c in data["cards"][:n]
            if c.get("front")
        ]
    return _heuristic_flashcards(chunks, n)


# ----------------------------- heuristics -----------------------------

_WORD = re.compile(r"[A-Za-z][A-Za-z-]+")


def _heuristic_mcqs(title: str, chunks: list[str], n: int) -> list[dict]:
    questions: list[dict] = []
    sentences = [s for chunk in chunks for s in split_sentences(chunk)]
    # prefer definition-ish sentences
    key_sentences = [s for s in sentences if re.search(r"\b(is|are|refers to|means|defined as)\b", s)]
    pool = (key_sentences + sentences)[:40]
    rng = random.Random(len(pool))
    vocab = list({w for s in sentences for w in _WORD.findall(s) if len(w) > 5})[:60]

    for s in pool:
        if len(questions) >= n:
            break
        words = _WORD.findall(s)
        candidates = [w for w in words if len(w) > 5 and w[0].isupper() or (w.lower() not in _STOPWORDS and len(w) > 6)]
        if not candidates:
            continue
        answer = rng.choice(candidates)
        stem = s.replace(answer, "______", 1)
        if "______" not in stem:
            continue
        distract_pool = [w for w in vocab if w != answer] or ["None"]
        distract = rng.sample(distract_pool, min(3, len(distract_pool)))
        options = distract + [answer]
        rng.shuffle(options)
        questions.append({
            "question": f"Fill in the blank: {stem}",
            "options": options,
            "answer_idx": options.index(answer),
            "explanation": f"From your material: {s[:180]}",
        })
    return questions[:n]


def _heuristic_flashcards(chunks: list[str], n: int) -> list[dict]:
    cards: list[dict] = []
    sentences = [s for chunk in chunks for s in split_sentences(chunk)]
    for s in sentences:
        if len(cards) >= n:
            break
        m = re.match(r"^(.{3,60}?)\s+(?:is|are|refers to|means)\s+(.{30,})", s)
        if m:
            front = f"What is {m.group(1).strip()}?"
            back = m.group(2).strip()
        else:
            words = _WORD.findall(s)
            key = [w for w in words if len(w) > 6]
            if not key:
                continue
            front = f"Explain the role of '{key[0]}' in your notes."
            back = s[:240]
        cards.append({"front": front, "back": back})
    return cards[:n]


_STOPWORDS = {
    "the", "and", "with", "that", "this", "from", "which", "their", "there", "these",
    "those", "have", "been", "were", "will", "would", "could", "should", "because",
}
