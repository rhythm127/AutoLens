import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from app.rag import HybridRetriever, VectorIndexService, chunk_pages, extract_pages
# NEW IMPORT: Bring in our Rule-Based Safety Engine
from app.safety import DiagnosticContext, SeverityEngine

app = FastAPI(
    title="AutoLens API",
    description="Backend API for AutoLens multimodal vehicle diagnosis platform.",
    version="0.2.0",
)

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
def create_vehicle(vehicle: VehicleCreate):
    """Receive vehicle details for troubleshooting."""
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

        vehicle_meta = {
            "manufacturer": manufacturer.strip(),
            "model": model.strip(),
            "year": year,
        }
        indexed_count = vector_service.index_chunks(chunks, vehicle_meta)

        return {
            "status": "indexed",
            "source": temp_pdf_path.name,
            "pages_extracted": len(pages),
            "chunks_indexed": indexed_count,
            "vehicle": vehicle_meta,
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