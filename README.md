# AutoLens

See the warning. Understand the problem. AutoLens combines your vehicle's PDF owner's manual with conversational troubleshooting, dashboard-image recognition, page citations, and the existing deterministic safety engine.

## Backend setup

Use Python 3.11 or newer. From the repository root in PowerShell:

```powershell
Copy-Item .env.example .env
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --env-file ../.env
```

On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` instead. Open [API documentation](http://127.0.0.1:8000/docs) or [health](http://127.0.0.1:8000/health). The first real ingestion/search loads the Sentence Transformers embedding model and may download its weights. Retrieval tests inject lightweight embeddings and do not need that download.

The `--env-file` option loads the root `.env`. Set either `OPENROUTER_API_KEY` or `GEMINI_API_KEY` for hosted chat and vision. When both are configured, OpenRouter is tried first, then Gemini. `OPENROUTER_MODEL` defaults to the PRD's `google/gemini-2.0-flash-exp:free`; model availability depends on the provider, so change it if unavailable. `OPENROUTER_VISION_MODEL` selects an image-capable model independently. `GEMINI_MODEL` defaults to `gemini-2.5-flash`.

With Ollama installed and running locally, `ollama pull llama3.2` enables the chat fallback at `http://localhost:11434/api/chat`. Set `OLLAMA_MODEL` to change the local model. If providers are unavailable, chat quotes the closest retrieved manual passage with its filename and page, followed by deterministic safety guidance and follow-up questions. No available passages produces an explicit lack-of-coverage response. Offline vision reports low confidence and asks for a clearer photo or text description; it never invents a detection.

`DATABASE_PATH` controls the SQLite file (default `backend/data/autolens.sqlite3`). `CHROMA_PERSIST_DIR` controls the separate vector store. Relative paths are resolved from the backend working directory. Keep both stores to preserve manuals and conversations. API credentials stay in the backend, never in frontend environment variables.

## Frontend setup

Use Node.js 22.12+ or 24+. In a second terminal from the repository root:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Use `npm` instead of `npm.cmd` on macOS/Linux. The `.cmd` spelling avoids PowerShell execution-policy restrictions on `npm.ps1`. Open [AutoLens](http://127.0.0.1:5173). Optional: copy `frontend/.env.example` to `frontend/.env.local` and change `VITE_API_URL` if the backend runs elsewhere. Restart Vite after changing it.

1. Enter a manufacturer, model, and year, then upload that vehicle's PDF manual.
2. Wait for the ready status, then describe the warning or upload a JPEG, PNG, or WebP dashboard image (maximum 10 MB).
3. A confident image result fills a question you can edit and send. Low-confidence results ask you to retake the photo or describe the symbol.
4. Read the reply, safety recommendation, and expandable page excerpts. Select a saved conversation to continue it. Start a new conversation before switching vehicles.

## API surface

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Health check |
| `POST /vehicles` | Validate and persist a vehicle, preserving the original response shape |
| `GET /vehicles` | List saved vehicle IDs and manual filenames |
| `POST /rag/ingest` | Multipart PDF, manufacturer, model, year; returns ingestion details and `vehicle_id` |
| `POST /rag/search` | Existing hybrid manual search |
| `POST /chat` | Message, optional session/vehicle ID or manufacturer/model/year, optional canonical warning type |
| `POST /vision/analyze` | Multipart `file`; returns `detected` or `low_confidence` |
| `GET /history/sessions?vehicle_id=...` | List saved sessions, optionally filtered by vehicle |
| `GET /history/sessions/{id}` | Full transcript with citations and severity per reply |
| `POST /diagnostics/evaluate` | Existing deterministic safety evaluation |
| `POST /diagnostics/session/start`, `/diagnostics/session/answer` | Existing diagnostic-agent endpoints |

Chat searches only the selected vehicle's manuals. A session retains its vehicle and prior user observations; mismatching vehicle details return 409. Missing session or vehicle IDs return 404. SQLite writes each user/assistant turn in one transaction. CORS permits `http://localhost:5173` and `http://127.0.0.1:5173`.

Hosted/local chat is prompted to use retrieved passages only, cite exact pages, and respect the severity engine. Model replies require matching evidence excerpts and inline citations; invalid evidence or explicit safety overrides fall back to the retrieved text. These checks are safeguards, not a guarantee that every model paraphrase is correct. The displayed severity and recommended action always come from the rule engine, never the model. This is a local development app without authentication or per-user access control; its saved histories are shared by clients of the same backend.

## Verification

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
cd ../frontend
npm.cmd run build
```

The original backend tests remain intact. `test_chat.py`, `test_vision.py`, and `test_history.py` cover grounded/offline responses, provider fallback, conversation context, canonical image validation, confidence thresholds, SQLite persistence, vehicle isolation, upload errors, and CORS. New tests inject providers and use temporary databases rather than live API calls. The frontend production bundle is written to `frontend/dist/`.
