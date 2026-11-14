# Multilingual Government Scheme Voice Agent — Integrated Backend

This repository combines Person 1's scheme knowledge/rule engine and Person 2's voice functions behind a stable FastAPI contract. Person 3's submitted application/dialogue files were empty stubs, so a minimal session/dialogue layer and HTTP API have been implemented here rather than importing empty files.

## Run locally / Replit

1. Install Python 3.11+ dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY` for voice transcription/translation.
3. Run: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. Open `/docs` for interactive API docs and `/api/health` for a health check.

The eligibility and scheme-list endpoints work without a Groq key. Voice transcription and translation require `GROQ_API_KEY`; TTS uses gTTS and requires internet access.

## API contract for the future Replit UI

- `GET /api/health`
- `GET /api/languages`
- `GET /api/schemes`
- `GET /api/schemes/{scheme_id}`
- `GET /api/schemes/{scheme_id}/context?query=...` (lazy-loads multilingual retrieval model)
- `POST /api/session` body/query: `language=en`
- `POST /api/chat` JSON: `{ "session_id": "...", "message": "...", "language": "en", "profile": {"age": 32, "state": "Karnataka"} }`
- `POST /api/eligibility` JSON: `{ "profile": {"age": 32, "state": "Karnataka"} }`
- `POST /api/voice/transcribe` multipart: `file`, optional `language_hint`
- `POST /api/voice/translate` multipart: `text`, `src`, `tgt`
- `POST /api/voice/turn` multipart: `file`, optional `language`, `session_id`, `profile_json`; runs STT → translation → dialogue → translated TTS in one request, returning audio as base64
- `POST /api/voice/speak` JSON: `{ "text": "...", "language": "kn" }`, returns MP3 bytes
- `DELETE /api/session/{session_id}`

## Frontend integration

Set `API_BASE_URL` in the Replit frontend to this backend's public URL. Use JSON for normal endpoints and multipart `FormData` for audio upload. Do not set the multipart Content-Type header manually; the browser adds the boundary.

## Important implementation notes

- Person 1's rule engine remains the source of eligibility decisions; no LLM is used to decide eligibility.
- User sessions are in-memory and ephemeral; they reset when the server restarts and are not suitable for multi-instance production deployment.
- No Aadhaar, bank account, phone number, or OTP fields are collected.
- Scheme records and citations are taken from Person 1's local JSON files. Verify source links and dates before public deployment.
- Current dialogue layer collects profile values when the frontend sends a structured `profile` patch. It does not yet reliably extract arbitrary values from free-form speech; wire a validated extraction step into `/api/chat` before claiming fully natural profile extraction.
- Person 1's vector retrieval module is retained and exposed through `/api/schemes/{scheme_id}/context`; it is lazy-loaded so API startup does not download an embedding model. The first retrieval call may require downloading the multilingual embedding model.
