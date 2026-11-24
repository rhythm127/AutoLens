from dotenv import load_dotenv

import tempfile
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR.parent / ".env", override=True)
load_dotenv(BACKEND_DIR / ".env", override=True)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.rag import HybridRetriever, VectorIndexService, chunk_pages, extract_pages
# NEW IMPORT: Bring in our Rule-Based Safety Engine
from app.safety import DiagnosticContext, SeverityEngine
from app.db.database import Database
from app.chat.chat_service import ChatService, CANONICAL_WARNINGS
from app.vision.analyzer import VisionAnalyzer
from uuid import uuid4

app = FastAPI(
    title="AutoLens API",
    description="Backend API for AutoLens multimodal vehicle diagnosis platform.",
    version="0.2.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def get_database():
    return Database()


def get_vision_analyzer():
    return VisionAnalyzer()

# Initialize local ChromaDB service, Hybrid Retriever, and Severity Engine
vector_service = VectorIndexService()
hybrid_retriever = HybridRetriever(vector_service=vector_service)
severity_engine = SeverityEngine()  # NEW: Create our safety rule checker


# --- Schemas (Input Validation Boxes) --------------------------------------

class VehicleCreate(BaseModel):
    manufacturer: str = Field(..., min_length=1, max_length=50)
    model: str = Field(..., min_length=1, max_length=50)
    year: int = Field(..., ge=1980, le=2027)

    @field_validator("manufacturer", "model", mode="before")
    @classmethod
    def strip_whitespace(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("Value cannot be empty or whitespace-only")
        return value


class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Warning light or fault code query")
    n_results: int = Field(default=5, ge=1, le=20)
    manufacturer: str | None = None
    model: str | None = None
    year: int | None = None


# NEW SCHEMA: Defines what JSON data the user sends to /diagnostics/evaluate
class SeverityEvaluateRequest(BaseModel):
    warning_type: str = Field(..., min_length=1, description="Identified warning symbol or fault")
    light_state: str = Field(default="continuous", description="'continuous' or 'flashing'")
    driving_behaviour: str = Field(default="normal", description="'normal' or symptom like 'shaking'")
    unusual_noise: bool = Field(default=False)
    manufacturer: str | None = None
    model: str | None = None
    year: int | None = None


# --- Endpoints (URLs the frontend can call) --------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint to verify backend operational status."""
    return {
        "status": "ok",
        "app": "AutoLens",
    }


@app.post("/vehicles", status_code=200)
def create_vehicle(vehicle: VehicleCreate, database: Database = Depends(get_database)):
    """Receive vehicle details for troubleshooting."""
    database.save_vehicle(vehicle.manufacturer, vehicle.model, vehicle.year)
    return {
        "status": "received",
        "vehicle": {
            "manufacturer": vehicle.manufacturer,
            "model": vehicle.model,
            "year": vehicle.year,
        },
    }


@app.post("/rag/ingest", status_code=200)
async def ingest_manual(
    file: UploadFile = File(...),
    manufacturer: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    database: Database = Depends(get_database),
):
    """
    Upload a vehicle owner's manual PDF -> Extract pages -> Chunk text -> Index in ChromaDB.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .pdf")

    temp_dir = Path(tempfile.mkdtemp())
    temp_pdf_path = temp_dir / Path(file.filename).name

    try:
        content = await file.read()
        temp_pdf_path.write_bytes(content)

        pages = extract_pages(temp_pdf_path)
        chunks = chunk_pages(pages, chunk_size=500, overlap=100)

        vehicle_meta = VehicleCreate(manufacturer=manufacturer, model=model, year=year).model_dump()
        indexed_count = vector_service.index_chunks(chunks, vehicle_meta)
        saved_vehicle = database.save_vehicle(**vehicle_meta, manual_filename=temp_pdf_path.name)

        return {
            "status": "indexed",
            "source": temp_pdf_path.name,
            "pages_extracted": len(pages),
            "chunks_indexed": indexed_count,
            "vehicle": vehicle_meta,
            "vehicle_id": saved_vehicle["id"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_pdf_path.exists():
            temp_pdf_path.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()


@app.post("/rag/search", status_code=200)
def search_manuals(request: HybridSearchRequest):
    """
    Query indexed manuals using Hybrid RAG (ChromaDB Dense + BM25 Sparse + RRF).
    """
    conditions: list[dict[str, Any]] = []
    if request.manufacturer:
        conditions.append({"manufacturer": request.manufacturer.strip()})
    if request.model:
        conditions.append({"model": request.model.strip()})
    if request.year is not None:
        conditions.append({"year": request.year})

    where_filter: dict[str, Any] | None = None
    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    results = hybrid_retriever.search(
        query_text=request.query,
        n_results=request.n_results,
        where=where_filter,
    )

    return {
        "query": request.query,
        "count": len(results),
        "results": results,
    }


# NEW ENDPOINT: Combines RAG Search + Rule-Based Severity Engine
@app.post("/diagnostics/evaluate", status_code=200)
def evaluate_diagnostics(request: SeverityEvaluateRequest):
    """
    1. Searches the vehicle manual for the warning light.
    2. Passes the manual text + user symptoms into SeverityEngine.
    3. Returns LOW/MEDIUM/HIGH severity and manual page citations.
    """
    # Build filter for manufacturer/model/year (same as /rag/search above)
    conditions: list[dict[str, Any]] = []
    if request.manufacturer:
        conditions.append({"manufacturer": request.manufacturer.strip()})
    if request.model:
        conditions.append({"model": request.model.strip()})
    if request.year is not None:
        conditions.append({"year": request.year})

    where_filter: dict[str, Any] | None = None
    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    # Step A: Search the manual for this warning light
    retrieved_records = hybrid_retriever.search(
        query_text=request.warning_type.replace("_", " "),
        n_results=3,
        where=where_filter,
    )
    # List comprehension: grab just the "text" string from each dictionary
    manual_texts = [r["text"] for r in retrieved_records]

    # Step B: Run our if/else safety rules
    try:
        context = DiagnosticContext(
            warning_type=request.warning_type,
            light_state=request.light_state,
            driving_behaviour=request.driving_behaviour,
            unusual_noise=request.unusual_noise,
            retrieved_manual_texts=manual_texts,
        )
        assessment = severity_engine.evaluate(context)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Step C: Return the final dictionary (FastAPI turns this into JSON automatically)
    return {
        "warning_type": request.warning_type,
        "assessment": assessment.to_dict(),
        "citations": [
            {
                "source": r["source"],
                "page_number": r["page_number"],
                "text": r["text"],
            }
            for r in retrieved_records
        ],
    }


# Add to imports at the top of backend/app/main.py:
from app.agents import DiagnosticAgent

# Initialize alongside severity_engine:
diagnostic_agent = DiagnosticAgent()

# Add Schemas:
class SessionStartRequest(BaseModel):
    warning_type: str = Field(..., min_length=1)
    manufacturer: str | None = None
    model: str | None = None
    year: int | None = None
    light_state: str | None = None
    driving_behaviour: str | None = None
    unusual_noise: bool | None = None

class SessionAnswerRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)

# Add Endpoints:
@app.post("/diagnostics/session/start", status_code=200)
def start_diagnostic_session(request: SessionStartRequest):
    """Start an interactive diagnostic chat session and return the first follow-up question."""
    try:
        state = diagnostic_agent.start_session(
            warning_type=request.warning_type,
            manufacturer=request.manufacturer,
            model=request.model,
            year=request.year,
            light_state=request.light_state,
            driving_behaviour=request.driving_behaviour,
            unusual_noise=request.unusual_noise,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "state": state.to_dict(),
        "next_question": diagnostic_agent.get_next_question(state),
    }

@app.post("/diagnostics/session/answer", status_code=200)
def answer_diagnostic_session(request: SessionAnswerRequest):
    """
    Record the user's answer to the current follow-up question.
    If all slots are filled, automatically run RAG + SeverityEngine and return the final assessment!
    """
    try:
        state = diagnostic_agent.process_answer(
            session_id=request.session_id,
            user_message=request.answer,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not state.is_complete:
        return {
            "status": "in_progress",
            "state": state.to_dict(),
            "next_question": diagnostic_agent.get_next_question(state),
            "result": None,
        }

    # All slots collected -> Run Hybrid RAG + SeverityEngine automatically
    eval_req = SeverityEvaluateRequest(
        warning_type=state.warning_type,
        light_state=state.light_state or "continuous",
        driving_behaviour=state.driving_behaviour or "normal",
        unusual_noise=bool(state.unusual_noise),
        manufacturer=state.manufacturer,
        model=state.model,
        year=state.year,
    )
    final_evaluation = evaluate_diagnostics(eval_req)
    return {
        "status": "complete",
        "state": state.to_dict(),
        "next_question": None,
        "result": final_evaluation,
    }


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, min_length=1, max_length=100)
    vehicle_id: str | None = Field(default=None, min_length=1, max_length=100)
    manufacturer: str | None = Field(default=None, min_length=1, max_length=50)
    model: str | None = Field(default=None, min_length=1, max_length=50)
    year: int | None = Field(default=None, ge=1980, le=2027)
    message: str = Field(min_length=1, max_length=8000)
    warning_type: str | None = None

    @field_validator("message", "manufacturer", "model", mode="before")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("warning_type")
    @classmethod
    def validate_warning(cls, value):
        if value is not None and value not in CANONICAL_WARNINGS:
            raise ValueError("Unknown canonical warning type")
        return value


@app.get("/vehicles")
def list_vehicles(database: Database = Depends(get_database)):
    return {"vehicles": database.vehicles()}


@app.post("/chat")
def chat(request: ChatRequest, database: Database = Depends(get_database)):
    session = database.session(request.session_id) if request.session_id else None
    if request.session_id and session is None:
        raise HTTPException(404, "Session not found")
    vehicle_id = request.vehicle_id or (session and session["vehicle_id"])
    vehicle = database.vehicle(vehicle_id) if vehicle_id else None
    if vehicle_id and vehicle is None:
        raise HTTPException(404, "Vehicle not found")
    if session and request.vehicle_id and request.vehicle_id != session["vehicle_id"]:
        raise HTTPException(409, "Start a new session to change vehicles")
    details = [request.manufacturer, request.model, request.year]
    if any(value is not None for value in details):
        if not all(value is not None for value in details):
            raise HTTPException(422, "Supply manufacturer, model and year together")
        if vehicle and any(vehicle[k] != getattr(request, k) for k in ("manufacturer", "model", "year")):
            raise HTTPException(409, "Vehicle details do not match this session or vehicle ID")
        if not vehicle:
            if session:
                raise HTTPException(409, "Start a new session to select a vehicle")
            vehicle = database.save_vehicle(*details)
            vehicle_id = vehicle["id"]
    result, warning = ChatService(hybrid_retriever, severity_engine).respond(
        request.message, vehicle, session["messages"] if session else [],
        request.warning_type, session["warning_type"] if session else None)
    session_id = request.session_id or str(uuid4())
    database.save_turn(session_id, vehicle_id, warning, request.message, result)
    return {"session_id": session_id, **result}


@app.post("/vision/analyze")
def analyze_dashboard(file: UploadFile = File(...), analyzer: VisionAnalyzer = Depends(get_vision_analyzer)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Upload a JPEG, PNG or WebP image")
    data = file.file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image must be 10 MB or smaller")
    valid = ((file.content_type == "image/jpeg" and data.startswith(b"\xff\xd8\xff")) or
             (file.content_type == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n")) or
             (file.content_type == "image/webp" and data.startswith(b"RIFF") and data[8:12] == b"WEBP"))
    if not valid:
        raise HTTPException(400, "Image contents do not match the declared format")
    return analyzer.analyze(data, file.content_type)


@app.get("/history/sessions")
def list_sessions(vehicle_id: str | None = None, database: Database = Depends(get_database)):
    return {"sessions": database.sessions(vehicle_id)}


@app.get("/history/sessions/{session_id}")
def session_history(session_id: str, database: Database = Depends(get_database)):
    session = database.session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session
