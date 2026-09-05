"""Quick check: what do generated MCQs look like (heur + LLM)?"""
from pathlib import Path

from app.services import llm, parser, quiz_gen


def show(tag, questions):
    print(f"\n===== {tag} =====")
    for i, q in enumerate(questions, 1):
        print(f"Q{i}) {q['question']}")
        for oi, opt in enumerate(q["options"]):
            mark = "  <-- correct" if oi == q["answer_idx"] else ""
            print(f"   {oi}) {opt}{mark}")
        if q.get("explanation"):
            print(f"   EXPL: {q['explanation'][:160]}")


def main() -> None:
    chunks = [t for _, t in parser.extract_chunks(Path("test_sample/ml_notes.txt"))]

    show("OFFLINE / HEURISTIC FALLBACK", quiz_gen._heuristic_mcqs("ML notes", chunks, 5))

    if llm.llm_available():
        print("\n>>> Gemini is available — calling generate_mcqs (will use the LLM)…")
        show("GEMINI (generate_mcqs)", quiz_gen.generate_mcqs("ML notes", chunks, 5))
    else:
        print("\n>>> No API key configured — LLM path skipped (only heuristic shown above).")


if __name__ == "__main__":
    main()