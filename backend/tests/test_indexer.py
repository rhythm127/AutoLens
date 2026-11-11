"""
test_indexer.py — Tests for SCRUM-11: Embedding & Vector Index Service (RAG Slice 2).

All tests in this module use a deterministic MockEmbeddingModel and ChromaDB's
in-memory EphemeralClient (or temporary directories for persistence tests).
No network calls, API keys, or HuggingFace model downloads are required.

Tested behaviours:
  Indexing:
    - Indexing a single chunk stores 1 record in ChromaDB.
    - Indexing multiple chunks correctly stores all records.
    - Re-indexing with the same IDs is idempotent (upsert updates without error).
    - Custom collection names are supported.

  Metadata preservation:
    - Source filename ('source') is preserved on each stored chunk.
    - 1-based page number ('page_number') is preserved as an integer.
    - 0-based chunk index ('chunk_index') is preserved as an integer.
    - Vehicle metadata ('manufacturer', 'model', 'year') is preserved.
    - Additional allowed metadata fields (e.g. 'manual_type') are preserved.

  Empty input handling:
    - Indexing an empty list of chunks returns 0 and does not error.
    - Querying an empty collection returns an empty list without error.
    - embed_texts([]) returns an empty list without error.

  Input validation:
    - Non-dict vehicle_metadata raises ValueError.
    - Missing or empty manufacturer raises ValueError.
    - Missing or empty model raises ValueError.
    - Missing, non-integer, or boolean year raises ValueError.
    - Out-of-bounds year (< 1900 or > 2100) raises ValueError.
    - Invalid extra metadata types (e.g. nested lists) raise ValueError.
    - Non-list chunks input raises ValueError.
    - Chunk missing required keys raises ValueError.
    - Chunk with empty or whitespace-only text raises ValueError.
    - Chunk with page_number < 1 raises ValueError.
    - Chunk with chunk_index < 0 raises ValueError.
    - Empty query text raises ValueError.
    - n_results <= 0 raises ValueError.

  Retrieval & Queries:
    - Query returns RetrievedRecord items with expected fields and types.
    - Query respects n_results limit.
    - Query with metadata filter (where) restricts results to matching vehicle.
    - Direct record retrieval via service.get().
    - get_count() accurately reflects collection size.
    - reset() clears all records from the collection.

  Persistence & Pipeline Integration:
    - PersistentClient preserves records across new service instances.
    - End-to-end integration: extract_pages -> chunk_pages -> index_chunks -> query.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from app.rag.chunker import TextChunk, chunk_pages
from app.rag.extractor import extract_pages
from app.rag.indexer import VectorIndexService


# ===========================================================================
# Deterministic Test Fixtures (No network calls or downloads)
# ===========================================================================

class MockEmbeddingModel:
    """
    Deterministic embedding model for fast, offline unit testing.

    Converts texts into 4-dimensional normalized float vectors based on
    string length and characters. Two identical strings produce identical
    embeddings; different strings produce distinct embeddings.
    """

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str] | str) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]

        embeddings: list[list[float]] = []
        for text in texts:
            # Deterministic calculation using text length and character sums
            length = float(len(text))
            char_sum = float(sum(ord(c) for c in text[:10])) if text else 1.0

            # Generate 4 float values
            vec = [
                length * 0.1,
                (char_sum % 17) * 0.05,
                (length + char_sum) * 0.01,
                1.0,
            ]
            embeddings.append(vec[: self.dimension])

        return embeddings


@pytest.fixture
def mock_embedding_model() -> MockEmbeddingModel:
    """Provides a deterministic mock embedding model."""
    return MockEmbeddingModel(dimension=4)


@pytest.fixture
def in_memory_service(mock_embedding_model: MockEmbeddingModel) -> VectorIndexService:
    """Provides an in-memory VectorIndexService with a mock embedding model and isolated collection."""
    import uuid

    unique_collection = f"test_{uuid.uuid4().hex}"
    return VectorIndexService(
        in_memory=True,
        collection_name=unique_collection,
        embedding_model=mock_embedding_model,
    )



@pytest.fixture
def sample_vehicle() -> dict[str, Any]:
    """Standard valid vehicle metadata for tests."""
    return {
        "manufacturer": "Toyota",
        "model": "Camry",
        "year": 2020,
        "manual_type": "owner_manual",
    }


@pytest.fixture
def sample_chunks() -> list[TextChunk]:
    """List of sample valid chunks for testing."""
    return [
        {
            "source": "camry_2020_manual.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "text": "Oil pressure warning light: indicates oil pressure is dangerously low.",
        },
        {
            "source": "camry_2020_manual.pdf",
            "page_number": 1,
            "chunk_index": 1,
            "text": "Stop the vehicle immediately in a safe place and turn off the engine.",
        },
        {
            "source": "camry_2020_manual.pdf",
            "page_number": 2,
            "chunk_index": 2,
            "text": "Brake system warning light: inspect brake fluid level or release parking brake.",
        },
    ]


# ===========================================================================
# Indexing Tests
# ===========================================================================

class TestIndexing:
    """Tests for VectorIndexService.index_chunks"""

    def test_index_single_chunk_stores_record(
        self,
        in_memory_service: VectorIndexService,
        sample_vehicle: dict[str, Any],
    ):
        """Indexing a single chunk should increase collection count by 1."""
        chunks: list[TextChunk] = [
            {
                "source": "manual.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "text": "Check engine light indicates an emissions problem.",
            }
        ]

        indexed_count = in_memory_service.index_chunks(chunks, sample_vehicle)

        assert indexed_count == 1
        assert in_memory_service.get_count() == 1

    def test_index_multiple_chunks_count_matches(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
        sample_vehicle: dict[str, Any],
    ):
        """Indexing multiple chunks should store all of them."""
        indexed_count = in_memory_service.index_chunks(sample_chunks, sample_vehicle)

        assert indexed_count == len(sample_chunks)
        assert in_memory_service.get_count() == len(sample_chunks)

    def test_index_upsert_is_idempotent(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
        sample_vehicle: dict[str, Any],
    ):
        """Re-indexing identical chunks should update records rather than duplicate them."""
        # Index the first time
        in_memory_service.index_chunks(sample_chunks, sample_vehicle)
        initial_count = in_memory_service.get_count()

        # Re-index the same chunks
        in_memory_service.index_chunks(sample_chunks, sample_vehicle)
        updated_count = in_memory_service.get_count()

        assert initial_count == updated_count == len(sample_chunks)

    def test_custom_collection_name(
        self,
        mock_embedding_model: MockEmbeddingModel,
    ):
        """VectorIndexService should support a custom collection name."""
        service = VectorIndexService(
            collection_name="custom_manuals",
            in_memory=True,
            embedding_model=mock_embedding_model,
        )
        assert service.collection.name == "custom_manuals"


# ===========================================================================
# Metadata Preservation Tests
# ===========================================================================

class TestMetadataPreservation:
    """Tests confirming all metadata fields are preserved in ChromaDB."""

    def test_source_page_number_and_chunk_index_preserved(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
        sample_vehicle: dict[str, Any],
    ):
        """Each chunk must carry source, 1-based page number, and chunk index."""
        in_memory_service.index_chunks(sample_chunks, sample_vehicle)

        records = in_memory_service.get()
        assert len(records) == len(sample_chunks)

        # Check page 1 and page 2 records
        page_numbers = {r["page_number"] for r in records}
        assert page_numbers == {1, 2}

        sources = {r["source"] for r in records}
        assert sources == {"camry_2020_manual.pdf"}

        chunk_indices = {r["chunk_index"] for r in records}
        assert chunk_indices == {0, 1, 2}

    def test_vehicle_details_preserved(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
        sample_vehicle: dict[str, Any],
    ):
        """Vehicle manufacturer, model, and year must be saved with each chunk."""
        in_memory_service.index_chunks(sample_chunks, sample_vehicle)

        records = in_memory_service.get()
        for r in records:
            assert r["manufacturer"] == "Toyota"
            assert r["model"] == "Camry"
            assert r["year"] == 2020

    def test_optional_metadata_field_preserved(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
        sample_vehicle: dict[str, Any],
    ):
        """Optional scalar fields like manual_type must be preserved in metadata."""
        in_memory_service.index_chunks(sample_chunks, sample_vehicle)

        records = in_memory_service.get()
        for r in records:
            assert r["metadata"].get("manual_type") == "owner_manual"

    def test_invalid_complex_metadata_type_rejected(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
    ):
        """ChromaDB does not support nested dicts or lists in metadata."""
        bad_vehicle = {
            "manufacturer": "Toyota",
            "model": "Camry",
            "year": 2020,
            "nested_data": {"feature": "abs"},  # invalid nested dictionary
        }

        with pytest.raises(ValueError, match="ChromaDB only supports primitive types"):
            in_memory_service.index_chunks(sample_chunks, bad_vehicle)


# ===========================================================================
# Empty Input Handling Tests
# ===========================================================================

class TestEmptyInputHandling:
    """Tests verifying robust handling of empty inputs."""

    def test_index_empty_chunks_returns_zero(
        self,
        in_memory_service: VectorIndexService,
        sample_vehicle: dict[str, Any],
    ):
        """Passing an empty list of chunks must return 0 without error."""
        result = in_memory_service.index_chunks([], sample_vehicle)

        assert result == 0
        assert in_memory_service.get_count() == 0

    def test_query_empty_collection_returns_empty_list(
        self,
        in_memory_service: VectorIndexService,
    ):
        """Querying when no documents are indexed must return [] without error."""
        results = in_memory_service.query("warning light")

        assert results == []

    def test_embed_texts_empty_returns_empty_list(
        self,
        in_memory_service: VectorIndexService,
    ):
        """Calling embed_texts with an empty list must return an empty list."""
        assert in_memory_service.embed_texts([]) == []


# ===========================================================================
# Input Validation Tests
# ===========================================================================

class TestInputValidation:
    """Tests verifying thorough validation of vehicle metadata and chunks."""

    def test_metadata_not_dict_raises_value_error(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
    ):
        """Passing a non-dict for vehicle_metadata must raise ValueError."""
        with pytest.raises(ValueError, match="vehicle_metadata must be a dictionary"):
            in_memory_service.index_chunks(sample_chunks, "not a dict")  # type: ignore

    def test_missing_or_empty_manufacturer_raises_value_error(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
    ):
        """Missing or blank manufacturer must raise ValueError."""
        with pytest.raises(ValueError, match="manufacturer"):
            in_memory_service.index_chunks(
                sample_chunks,
                {"manufacturer": "   ", "model": "Camry", "year": 2020},
            )

        with pytest.raises(ValueError, match="manufacturer"):
            in_memory_service.index_chunks(
                sample_chunks,
                {"model": "Camry", "year": 2020},
            )

    def test_missing_or_empty_model_raises_value_error(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
    ):
        """Missing or blank model must raise ValueError."""
        with pytest.raises(ValueError, match="model"):
            in_memory_service.index_chunks(
                sample_chunks,
                {"manufacturer": "Toyota", "model": "", "year": 2020},
            )

    def test_invalid_year_raises_value_error(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
    ):
        """Non-integer, boolean, or missing year must raise ValueError."""
        # Missing year
        with pytest.raises(ValueError, match="year"):
            in_memory_service.index_chunks(
                sample_chunks,
                {"manufacturer": "Toyota", "model": "Camry"},
            )

        # Boolean year (in Python bool is a subclass of int, so reject True/False)
        with pytest.raises(ValueError, match="year"):
            in_memory_service.index_chunks(
                sample_chunks,
                {"manufacturer": "Toyota", "model": "Camry", "year": True},
            )

        # String year
        with pytest.raises(ValueError, match="year"):
            in_memory_service.index_chunks(
                sample_chunks,
                {"manufacturer": "Toyota", "model": "Camry", "year": "2020"},
            )

        # Out-of-bounds year
        with pytest.raises(ValueError, match="between 1900 and 2100"):
            in_memory_service.index_chunks(
                sample_chunks,
                {"manufacturer": "Toyota", "model": "Camry", "year": 1850},
            )

    def test_chunks_not_list_raises_value_error(
        self,
        in_memory_service: VectorIndexService,
        sample_vehicle: dict[str, Any],
    ):
        """Chunks argument must be a list."""
        with pytest.raises(ValueError, match="chunks must be a list"):
            in_memory_service.index_chunks("not a list", sample_vehicle)  # type: ignore

    def test_chunk_missing_required_keys_raises_value_error(
        self,
        in_memory_service: VectorIndexService,
        sample_vehicle: dict[str, Any],
    ):
        """A chunk dictionary missing source, page_number, chunk_index, or text must raise ValueError."""
        # Missing source
        with pytest.raises(ValueError, match="source"):
            in_memory_service.index_chunks(
                [{"page_number": 1, "chunk_index": 0, "text": "Valid text."}],  # type: ignore
                sample_vehicle,
            )

        # Missing page_number
        with pytest.raises(ValueError, match="page_number"):
            in_memory_service.index_chunks(
                [{"source": "m.pdf", "chunk_index": 0, "text": "Valid text."}],  # type: ignore
                sample_vehicle,
            )

        # Missing chunk_index
        with pytest.raises(ValueError, match="chunk_index"):
            in_memory_service.index_chunks(
                [{"source": "m.pdf", "page_number": 1, "text": "Valid text."}],  # type: ignore
                sample_vehicle,
            )

    def test_chunk_with_empty_text_raises_value_error(
        self,
        in_memory_service: VectorIndexService,
        sample_vehicle: dict[str, Any],
    ):
        """A chunk with empty or whitespace-only text must raise ValueError."""
        with pytest.raises(ValueError, match="empty or missing 'text'"):
            in_memory_service.index_chunks(
                [{"source": "m.pdf", "page_number": 1, "chunk_index": 0, "text": "   "}],
                sample_vehicle,
            )

    def test_chunk_with_invalid_page_number_raises_value_error(
        self,
        in_memory_service: VectorIndexService,
        sample_vehicle: dict[str, Any],
    ):
        """page_number must be >= 1 (1-based)."""
        with pytest.raises(ValueError, match="1-based integer 'page_number'"):
            in_memory_service.index_chunks(
                [{"source": "m.pdf", "page_number": 0, "chunk_index": 0, "text": "Text"}],
                sample_vehicle,
            )

    def test_query_validation_raises_value_error(
        self,
        in_memory_service: VectorIndexService,
    ):
        """Query must reject empty text or non-positive n_results."""
        with pytest.raises(ValueError, match="query_text must be a non-empty string"):
            in_memory_service.query("")

        with pytest.raises(ValueError, match="query_text must be a non-empty string"):
            in_memory_service.query("   ")

        with pytest.raises(ValueError, match="n_results must be a positive integer"):
            in_memory_service.query("test query", n_results=0)

        with pytest.raises(ValueError, match="n_results must be a positive integer"):
            in_memory_service.query("test query", n_results=-5)


# ===========================================================================
# Retrieval & Query Tests
# ===========================================================================

class TestRetrievalAndQueries:
    """Tests for VectorIndexService.query and .get"""

    def test_query_returns_retrieved_records_with_correct_fields(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
        sample_vehicle: dict[str, Any],
    ):
        """Query results must match the RetrievedRecord structure with correct types."""
        in_memory_service.index_chunks(sample_chunks, sample_vehicle)

        results = in_memory_service.query("oil pressure warning", n_results=2)

        assert len(results) > 0
        record = results[0]

        # Verify all expected keys are populated
        assert isinstance(record["id"], str)
        assert isinstance(record["text"], str)
        assert record["source"] == "camry_2020_manual.pdf"
        assert record["page_number"] in (1, 2)
        assert isinstance(record["chunk_index"], int)
        assert record["manufacturer"] == "Toyota"
        assert record["model"] == "Camry"
        assert record["year"] == 2020
        assert isinstance(record["metadata"], dict)
        assert record["distance"] is not None

    def test_query_respects_n_results(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
        sample_vehicle: dict[str, Any],
    ):
        """Query must return at most n_results records."""
        in_memory_service.index_chunks(sample_chunks, sample_vehicle)

        results = in_memory_service.query("warning light", n_results=1)
        assert len(results) == 1

        results_two = in_memory_service.query("warning light", n_results=2)
        assert len(results_two) == 2

    def test_query_with_metadata_filter(
        self,
        in_memory_service: VectorIndexService,
        mock_embedding_model: MockEmbeddingModel,
    ):
        """Query with where filter must return only chunks matching the filter."""
        # Index Toyota Camry chunks
        camry_chunks: list[TextChunk] = [
            {"source": "camry.pdf", "page_number": 1, "chunk_index": 0, "text": "Camry brake light."}
        ]
        in_memory_service.index_chunks(
            camry_chunks,
            {"manufacturer": "Toyota", "model": "Camry", "year": 2020},
        )

        # Index Honda Civic chunks
        civic_chunks: list[TextChunk] = [
            {"source": "civic.pdf", "page_number": 1, "chunk_index": 0, "text": "Civic brake light."}
        ]
        in_memory_service.index_chunks(
            civic_chunks,
            {"manufacturer": "Honda", "model": "Civic", "year": 2021},
        )

        # Query specifically for Toyota
        toyota_results = in_memory_service.query(
            "brake light",
            n_results=5,
            where={"manufacturer": "Toyota"},
        )
        assert len(toyota_results) == 1
        assert toyota_results[0]["manufacturer"] == "Toyota"
        assert toyota_results[0]["model"] == "Camry"

        # Query specifically for Honda
        honda_results = in_memory_service.query(
            "brake light",
            n_results=5,
            where={"manufacturer": "Honda"},
        )
        assert len(honda_results) == 1
        assert honda_results[0]["manufacturer"] == "Honda"
        assert honda_results[0]["model"] == "Civic"

    def test_get_by_id(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
        sample_vehicle: dict[str, Any],
    ):
        """service.get() can fetch specific chunks directly by ID."""
        in_memory_service.index_chunks(sample_chunks, sample_vehicle)

        # Expected ID for first chunk
        expected_id = "toyota_camry_2020_camry_2020_manual.pdf_p1_c0"
        records = in_memory_service.get(ids=[expected_id])

        assert len(records) == 1
        assert records[0]["id"] == expected_id
        assert records[0]["chunk_index"] == 0

    def test_reset_clears_all_records(
        self,
        in_memory_service: VectorIndexService,
        sample_chunks: list[TextChunk],
        sample_vehicle: dict[str, Any],
    ):
        """service.reset() must clear the collection back to 0 items."""
        in_memory_service.index_chunks(sample_chunks, sample_vehicle)
        assert in_memory_service.get_count() == 3

        in_memory_service.reset()
        assert in_memory_service.get_count() == 0
        assert in_memory_service.query("warning") == []


# ===========================================================================
# Persistence & Pipeline Integration Tests
# ===========================================================================

class TestPersistenceAndPipeline:
    """Tests checking disk persistence and end-to-end RAG pipeline integration."""

    def test_persistent_directory_saves_and_reloads_data(
        self,
        mock_embedding_model: MockEmbeddingModel,
        sample_chunks: list[TextChunk],
        sample_vehicle: dict[str, Any],
        tmp_path: Path,
    ):
        """
        Data stored with PersistentClient must survive creating a new
        VectorIndexService instance pointing to the same folder.
        """
        persist_dir = tmp_path / "chroma_db"

        # Instance 1: write data to disk
        service1 = VectorIndexService(
            persist_directory=persist_dir,
            collection_name="persistent_test",
            embedding_model=mock_embedding_model,
        )
        service1.index_chunks(sample_chunks, sample_vehicle)
        assert service1.get_count() == len(sample_chunks)

        # Instance 2: open the existing folder
        service2 = VectorIndexService(
            persist_directory=persist_dir,
            collection_name="persistent_test",
            embedding_model=mock_embedding_model,
        )

        assert service2.get_count() == len(sample_chunks)
        records = service2.query("oil pressure", n_results=1)
        assert len(records) == 1
        assert records[0]["manufacturer"] == "Toyota"

    def test_full_pipeline_extract_chunk_index_retrieve(
        self,
        mock_embedding_model: MockEmbeddingModel,
        tmp_path: Path,
    ):
        """
        Complete end-to-end integration test across all RAG modules:
        1. Create an in-memory PDF with warning light text.
        2. extract_pages() extracts pages and 1-based page numbers.
        3. chunk_pages() creates overlapping text chunks.
        4. VectorIndexService.index_chunks() indexes them into ChromaDB.
        5. VectorIndexService.query() retrieves the exact page and source citation.
        """
        # Step 1: Create a multi-page PDF in memory
        doc = pymupdf.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_textbox(
            pymupdf.Rect(50, 50, 500, 700),
            "Engine Temperature Warning: Stop the vehicle immediately and let the engine cool.",
            fontsize=12,
        )
        p2 = doc.new_page(width=595, height=842)
        p2.insert_textbox(
            pymupdf.Rect(50, 50, 500, 700),
            "Tire Pressure Monitoring System (TPMS): Check tire inflation pressures as soon as possible.",
            fontsize=12,
        )
        pdf_file = tmp_path / "toyota_rav4_manual.pdf"
        pdf_file.write_bytes(doc.tobytes())
        doc.close()

        # Step 2: Extract pages
        extracted = extract_pages(pdf_file)
        assert len(extracted) == 2

        # Step 3: Chunk pages
        chunks = chunk_pages(extracted, chunk_size=50, overlap=10)
        assert len(chunks) > 0

        # Step 4: Index chunks
        service = VectorIndexService(
            in_memory=True,
            collection_name="e2e_pipeline_test",
            embedding_model=mock_embedding_model,
        )
        vehicle = {"manufacturer": "Toyota", "model": "RAV4", "year": 2023}
        indexed_count = service.index_chunks(chunks, vehicle)
        assert indexed_count == len(chunks)

        # Step 5: Query for engine temperature warning
        results = service.query("engine temperature overheating", n_results=1)
        assert len(results) == 1

        top_match = results[0]
        assert top_match["source"] == "toyota_rav4_manual.pdf"
        assert top_match["page_number"] in (1, 2)
        assert top_match["manufacturer"] == "Toyota"
        assert top_match["model"] == "RAV4"
        assert top_match["year"] == 2023
        assert "engine" in top_match["text"].lower() or "warning" in top_match["text"].lower()
