"""End-to-end smoke test against a running server (python test_smoke.py)."""
import io
import json
import time

import httpx

BASE = "http://127.0.0.1:8000"

client = httpx.Client(base_url=BASE, follow_redirects=False, timeout=120)

# 1. landing
r = client.get("/")
assert r.status_code == 200 and "StudyMate" in r.text, r.status_code
print("✓ landing page")

# 2. demo login -> dashboard
r = client.get("/auth/demo")
assert r.status_code in (302, 307) and r.headers["location"] == "/dashboard", (r.status_code, r.headers.get("location"))
r = client.get("/dashboard")
assert r.status_code == 200 and "Alex" in r.text
print("✓ demo login + dashboard")

# 3. upload
files = {"file": ("ml_notes.txt", io.BytesIO(open("test_sample/ml_notes.txt", "rb").read()), "text/plain")}
r = client.post("/api/documents/upload", files=files)
assert r.status_code == 200, r.text
doc_id = r.json()["id"]
print("✓ upload accepted, doc id", doc_id)

for _ in range(30):
    docs = client.get("/api/documents").json()
    doc = next(d for d in docs if d["id"] == doc_id)
    if doc["status"] != "processing":
        break
    time.sleep(0.5)
assert doc["status"] == "ready", doc
assert doc["summary"], "no summary generated"
print("✓ processing ready — summary:", doc["summary"][:90].replace("\n", " "), "…")
print("  topics:", doc["key_topics"])

# 4. viewer page
r = client.get(f"/documents/{doc_id}")
assert r.status_code == 200
print("✓ viewer page")

# 5. chat
r = client.post("/api/chat", json={"document_id": doc_id, "message": "What is overfitting and how do we reduce it?"})
data = r.json()
assert r.status_code == 200 and data["answer"], r.text
print("✓ chat answer:", data["answer"][:110].replace("\n", " "), "…")
print("  citations:", [(c["n"], c["label"]) for c in data["citations"]])

# 6. quiz generation (MCQ)
r = client.post("/api/quiz/generate", json={"document_id": doc_id, "kind": "mcq", "count": 5})
assert r.status_code == 200, r.text
quiz_id = r.json()["quiz_id"]
r = client.get(f"/quiz/{quiz_id}")
assert r.status_code == 200 and "Submit answers" in r.text
print("✓ MCQ generated + page renders")

# 7. submit quiz (all option 0)
r = client.post(f"/api/quiz/{quiz_id}/submit", json={"answers": [0, 0, 0, 0, 0]})
assert r.status_code == 200, r.text
print(f"✓ quiz submitted — score {r.json()['score']}/{r.json()['total']}")

# 8. flashcards
r = client.post("/api/quiz/generate", json={"document_id": doc_id, "kind": "flashcards", "count": 4})
assert r.status_code == 200, r.text
flash_id = r.json()["quiz_id"]
r = client.get(f"/quiz/{flash_id}")
assert r.status_code == 200 and "Flip card" in r.text
print("✓ flashcards generated + page renders")

# 9. planner
r = client.post("/api/tasks", json={"title": "Revise chapter 2", "course": "CS402", "due_date": "2026-09-10", "task_type": "reading"})
assert r.status_code == 200
task_id = r.json()["id"]
r = client.patch(f"/api/tasks/{task_id}", json={"status": "done"})
assert r.json()["status"] == "done"
assert client.get("/api/tasks").json()[0]["title"] == "Revise chapter 2"
assert client.get("/planner").status_code == 200
print("✓ planner create/toggle/page")

# 10. progress
r = client.get("/api/progress")
p = r.json()
assert r.status_code == 200 and p["documents"] >= 1 and p["avg_score"] >= 0
r = client.get("/progress")
assert r.status_code == 200
print("✓ progress API + page —", {k: p[k] for k in ("documents", "quiz_attempts", "avg_score", "tasks_done")})

# 11. settings + upload page render
assert client.get("/settings").status_code == 200
assert client.get("/upload").status_code == 200
print("✓ settings + upload pages")

# 12. unauth guard
anon = httpx.Client(base_url=BASE, follow_redirects=False)
assert anon.get("/dashboard").status_code in (302, 401)
print("✓ anonymous access redirected")

print("\nALL SMOKE TESTS PASSED ✅")
