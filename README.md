# StudyMate AI 🎓

AI academic assistant for your final year project: upload notes/syllabus/PDFs, get
summaries, ask grounded questions with citations, generate quizzes & flashcards,
plan study tasks and track progress.

## Core loop
**Ingest → Understand → Practice → Improve**

| Feature | How it works |
|---|---|
| 📄 Doc upload + summary | PDF / PPTX / DOCX / TXT / MD parsed, chunked, summarized (Gemini free tier, or extractive fallback) |
| 💬 RAG chatbot | TF-IDF retrieval over document chunks → LLM answers with [n] citations, or grounded excerpt fallback |
| 🧠 Quiz & flashcards | LLM-generated MCQs + flip flashcards, score tracking per attempt |
| 🗓️ Study planner | Tasks with course, due date, type; todo/done toggle |
| 📈 Progress dashboard | Topic mastery bars, weak-topic highlight, score history chart |

## Tech stack
- **Backend:** FastAPI + Jinja2 templates + Tailwind (CDN) — design tokens ported 1:1 from the Stitch design system
- **DB:** SQLite via SQLAlchemy (swap to Postgres via `DATABASE_URL`)
- **LLM:** Gemini, Mistral, or Groq (free tiers) with **automatic fallback** — set the primary via `LLM_PROVIDER`; any other provider with a key becomes a backup
- **Retrieval:** in-memory TF-IDF index (stdlib) — swap for Chroma/FAISS later, callers depend only on `ChunkHit`
- **Auth:** Google OAuth (authlib) with one-click demo login fallback

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # add GEMINI_API_KEY, MISTRAL_API_KEY, or GROQ_API_KEY (all free)
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 → **Try the live demo** (no OAuth needed).

### AI providers (all free tiers)
Provider keys go in `.env`. `LLM_PROVIDER` sets the primary; every provider that
has a key becomes an automatic fallback (e.g. `gemini` missing → tries `mistral` → `groq`).

| Provider | Key | Get one at |
|---|---|---|
| Gemini (default) | `GEMINI_API_KEY` | https://aistudio.google.com/apikey |
| Mistral | `MISTRAL_API_KEY` | https://console.mistral.ai |
| Groq | `GROQ_API_KEY` | https://console.groq.com/keys |

> Mistral's free tier is rate-limited; when it returns 429 the app automatically
> falls back to the next configured provider instead of erroring.

### Optional: real Google sign-in
1. [Google Cloud Console](https://console.cloud.google.com/apis/credentials) → Create credentials → **OAuth client ID** → Web application.
2. In the client's edit form, fill **two separate fields** (the #1 source of the
   `Invalid origin: URIs must not contain a path or end with "/"` error is putting the
   URL in the wrong box):
   - **Authorized JavaScript origins** → `http://127.0.0.1:8000`
     *(bare origin only — no path, no trailing slash)*
   - **Authorized redirect URIs** → `http://127.0.0.1:8000/auth/callback`
     *(this is the field that allows a path)*
3. Copy the Client ID + Client Secret into your `.env`:
   ```
   GOOGLE_CLIENT_ID=....apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=GOCSPX-....
   OAUTH_REDIRECT_BASE=http://127.0.0.1:8000
   ```
4. Restart the server and use **Sign in with Google**.

> The redirect URI the app uses is `${OAUTH_REDIRECT_BASE}/auth/callback` and must
> **exactly match** what you saved in Google (same scheme, host, port, path).

## Project layout
```
app/
├── main.py             # FastAPI app wiring
├── config.py           # env settings
├── database.py         # engine/session
├── models.py           # User, Document, Chunk, Quiz, ChatMessage, StudyTask…
├── auth.py             # session auth + OAuth helpers
├── routers/            # pages, auth, documents, chat, quiz, planner, progress
├── services/           # parser, llm, rag, quiz_gen, pipeline
├── static/             # css/js
└── templates/          # base, app shell, 8 pages
```

## Offline mode
No API key? Everything still runs: summaries are extractive, quizzes are
heuristic fill-in-blank/definition cards, chat returns cited excerpts. Add a key
whenever you're ready — no code changes.
