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
        f"Create {n} exam-style multiple-choice questions that genuinely test UNDERSTANDING "
        f"of this material — e.g. definitions, mechanisms, comparisons, consequences or "
        f"applications. Never use fill-in-the-blank. Each question must have exactly 4 "
        f"plausible options with one clearly correct answer.\n"
        'Respond as JSON: {"questions": [{"question": str, "options": [str, str, str, str], '
        '"answer_idx": 0-3, "explanation": str}]}'
    )
    data = llm.generate_json(prompt, system=SYSTEM_QUIZ, temperature=0.6)
    if data and isinstance(data, dict) and data.get("questions"):
        cleaned = []
        for q in data["questions"][:n]:
            stem = str(q.get("question") or "").strip()
            options = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()]
            # drop fill-in-the-blank / malformed items
            if not stem or len(options) < 3 or "____" in stem or "fill in the blank" in stem.lower():
                continue
            try:
                idx = int(q.get("answer_idx", -1))
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(options)):
                continue
            cleaned.append({
                "question": stem,
                "options": options,
                "answer_idx": idx,
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
    """Build real comprehension MCQs from note sentences (no fill-in-the-blank)."""
    sentences = [s for chunk in chunks for s in split_sentences(chunk)]
    cands = _extract_candidates(sentences)
    if not cands:
        return _vocab_fallback(sentences, n)

    rng = random.Random(hash(tuple(sentences)) & 0xFFFF)
    questions: list[dict] = []

    # 1) one question per concept, cycling through kinds/variants
    flip = 0
    for c in cands:
        if len(questions) >= n:
            break
        q = _build_question(c, cands, rng, flip)
        if q:
            questions.append(q)
        flip ^= 1

    # 2) if we still need more, add reverse "which concept matches X?" items
    if len(questions) < n:
        for c in cands:
            if len(questions) >= n:
                break
            if c["kind"] != "def":
                continue
            q = _build_reverse(c, cands, rng)
            if q:
                questions.append(q)
    return questions[:n]


def _clean_term(term: str) -> str:
    for marker in (" such as", " including", " e.g.", " like", " especially", " for example"):
        i = term.lower().find(marker)
        if i > 0:
            term = term[:i]
    term = term.strip(" :,.;")
    for marker in (" that", " which", " who", " whose", " when", " where"):
        if term.lower().endswith(marker):
            term = term[: -len(marker)]
            break
    term = re.sub(r"^(the|a|an)\s+", "", term, count=1, flags=re.I)
    return term.strip()


def _extract_candidates(sentences: list[str]) -> list[dict]:
    """Pull (term, kind, tail, sentence) pairs from note sentences."""
    patterns: list[tuple[re.Pattern, str]] = [
        (re.compile(r"^([A-Za-z][\w''\- ]{1,45}?)\s+(?:is|are)\s+(.+)$", re.I), "def"),
        (re.compile(r"^([A-Za-z][\w''\- ]{1,45}?)\s+(?:is\s+)?(?:defined as|referred to as)\s+(.+)$", re.I), "def"),
        (re.compile(r"^([A-Za-z][\w''\- ]{1,45}?)\s+refers to\s+(.+)$", re.I), "def"),
        (re.compile(r"^([A-Za-z][\w''\- ]{1,45}?)\s+occurs when\s+(.+)$", re.I), "when"),
        (re.compile(r"^([A-Za-z][\w''\- ]{1,45}?)\s+(?:is|are)\s+used (?:to|for)\s+(.+)$", re.I), "used"),
        (re.compile(r"^([A-Za-z][\w''\- ]{1,45}?)\s+measures\s+(.+)$", re.I), "measure"),
        (re.compile(r"^([A-Za-z][\w''\- ]{1,45}?)\s+(?:reduce[s]?|prevent[s]?)\s+(.+)$", re.I), "reduce"),
    ]
    cands: list[dict] = []
    seen: set[str] = set()
    _RELATIVE = {"that", "which", "who", "whose", "when", "where"}
    for s in sentences:
        for pat, kind in patterns:
            m = pat.match(s)
            if not m:
                continue
            raw_term = m.group(1)
            last_word = raw_term.split()[-1].strip(".,;:").lower()
            if last_word in _RELATIVE:
                continue  # "X that is ..." -> verb inside a relative clause, not a definition
            term = _clean_term(raw_term)
            tail = m.group(2).strip().strip(";:.,")
            if not term or len(tail) < 18 or len(term.split()) > 6:
                continue
            key = term.lower()
            if key in seen:
                break
            seen.add(key)
            cands.append({"term": term, "kind": kind, "tail": tail, "sentence": s})
            break
    return cands

def _distinct(pool: list[str], exclude: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for p in pool:
        pk = p.lower()
        if pk == exclude.lower() or pk in seen:
            continue
        seen.add(pk)
        out.append(p)
    return out


def _clip(text: str, n: int = 200) -> str:
    return text if len(text) <= n else text[:n].rstrip() + "…"


def _pick3(rng: random.Random, pool: list[str]) -> list[str] | None:
    pool = list(dict.fromkeys(pool))
    return rng.sample(pool, 3) if len(pool) >= 3 else None


def _make_options(rng: random.Random, correct: str, distract: list[str]) -> tuple[list[str], int]:
    options = [correct, *distract]
    rng.shuffle(options)
    return [_clip(o, 220) for o in options], options.index(correct)


def _build_question(c: dict, cands: list[dict], rng: random.Random, flip: bool) -> dict | None:
    kind, term, tail, sentence = c["kind"], c["term"], c["tail"], c["sentence"]
    if flip and kind == "def":
        return _build_reverse(c, cands, rng)
    if kind == "def":
        pool = _distinct([x["tail"] for x in cands if x is not c and x["kind"] == "def"], tail)
    elif kind in ("when", "used", "measure", "reduce"):
        pool = _distinct([x["tail"] for x in cands if x is not c], tail)
    else:
        return None
    distract = _pick3(rng, pool)
    if not distract:
        return None
    options, ans = _make_options(rng, tail, distract)
    stems = {
        "def": f"What is {term}?",
        "when": f"When does {term} occur?",
        "used": f"What is {term} used for?",
        "measure": f"What does {term} measure?",
        "reduce": f"According to your notes, what problem do {term} help to reduce?",
    }
    return {"question": stems[kind], "options": options, "answer_idx": ans,
            "explanation": f"From your notes: {_clip(sentence)}"}


def _build_reverse(c: dict, cands: list[dict], rng: random.Random) -> dict | None:
    pool = _distinct([x["term"] for x in cands if x is not c], c["term"])
    distract = _pick3(rng, pool)
    if not distract:
        return None
    options, ans = _make_options(rng, c["term"], distract)
    return {"question": f"Which concept is described as: “{_clip(c['tail'], 160)}”?",
            "options": options, "answer_idx": ans,
            "explanation": f"From your notes: {_clip(c['sentence'])}"}


def _vocab_fallback(sentences: list[str], n: int) -> list[dict]:
    """Last resort: definition questions when no sentence-start patterns match."""
    questions: list[dict] = []
    for s in sentences:
        if len(questions) >= n:
            break
        m = re.match(r"^(.{3,60}?)\s+(?:is|are|refers to|means)\s+(.{30,})", s)
        if m:
            term, rest = m.group(1).strip(), m.group(2).strip()
            correct = _clip(rest, 180)
            options = [correct, "The opposite is emphasized in your notes.",
                       "It is unrelated to the material.", "None of the above."]
            rng = random.Random(hash(s) & 0xFFFF)
            rng.shuffle(options)
            questions.append({
                "question": f"What is {term}?", "options": options,
                "answer_idx": options.index(correct),
                "explanation": f"From your notes: {_clip(s)}",
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
