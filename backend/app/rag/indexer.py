"""
indexer.py — Embedding and Vector Index Service for AutoLens RAG pipeline.

This module is responsible for:
1. Converting text chunks from vehicle owner's manuals into mathematical
   vectors (embeddings) using Sentence Transformers.
2. Storing those embeddings, the original chunk text, and rich metadata
   (source PDF filename, 1-based page number, chunk index, vehicle make/model/year)
   inside ChromaDB, a lightweight, local vector database.
3. Querying stored chunks by semantic similarity so future diagnostic steps
   can retrieve the exact manual paragraph and page number for any warning light.

No external cloud services or paid API keys are used. Everything runs locally.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

import chromadb
from chromadb.api.models.Collection import Collection

from app.rag.chunker import TextChunk


# ---------------------------------------------------------------------------
# Default Configuration Constants
# ---------------------------------------------------------------------------

# Lightweight, free, open-source model running locally on CPU.
# Produces 384-dimensional dense vector embeddings.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Default ChromaDB collection name for vehicle manuals
DEFAULT_COLLECTION_NAME = "autolens_manuals"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class VehicleMetadata(TypedDict, total=False):
    """
    Standard vehicle metadata required for indexing manual chunks.

    Attributes
    ----------
    manufacturer : str
        Car manufacturer / make (e.g. 'Toyota', 'Honda').
    model : str
        Car model name (e.g. 'Camry', 'Civic').
    year : int
        Vehicle model year (e.g. 2020).
    manual_type : str, optional
        Optional descriptor for document type (e.g. 'owner_manual').
    """

    manufacturer: str
    model: str
    year: int
    manual_type: str


class RetrievedRecord(TypedDict):
    """
    Structured search result returned from the vector index.

    Attributes
    ----------
    id : str
        The unique ID of the chunk in ChromaDB.
    text : str
        The original text of the manual chunk.
    source : str
        The filename of the PDF manual (e.g. 'camry_2020.pdf').
    page_number : int
        The 1-based page number where this text appears in the manual.
    chunk_index : int
        The 0-based chunk position in the document.
    manufacturer : str
        Vehicle manufacturer (e.g. 'Toyota').
    model : str
        Vehicle model (e.g. 'Camry').
    year : int
        Vehicle model year (e.g. 2020).
    metadata : dict[str, Any]
        The complete metadata dictionary stored in ChromaDB.
    distance : float | None
        Cosine distance from query embedding (lower means more similar).
    """

    id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    manufacturer: str
    model: str
    year: int
    metadata: dict[str, Any]
    distance: float | None


# ---------------------------------------------------------------------------
# Vector Index Service
# ---------------------------------------------------------------------------

class VectorIndexService:
    """
    Service for converting manual chunks into embeddings and indexing them in ChromaDB.

    Parameters
    ----------
    persist_directory : str or Path or None
        Local filesystem directory where ChromaDB stores its persistent files.
        If None, the service checks the CHROMA_PERSIST_DIR environment variable.
        If still unset, it defaults to 'backend/data/chroma'.
    collection_name : str
        Name of the ChromaDB collection to store manual chunks in.
    embedding_model : Any, optional
        An optional custom or mock embedding model with an `.encode(texts)` method.
        When provided, this model is used directly without loading SentenceTransformer.
        This allows unit tests to run fast without downloading weights from the web.
    model_name : str
        The SentenceTransformer model name to use when embedding_model is None.
    in_memory : bool
        If True, runs ChromaDB purely in memory (EphemeralClient). Useful for
        tests that should not write anything to disk.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: Any = None,
        model_name: str = DEFAULT_MODEL_NAME,
        in_memory: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self._model_name = model_name
        self._embedding_model = embedding_model
        self.in_memory = in_memory

        # Setup ChromaDB client based on requested persistence mode
        if in_memory or persist_directory == ":memory:":
            self.persist_directory: Path | None = None
            # EphemeralClient keeps everything in RAM — ideal for unit tests
            self.client = chromadb.EphemeralClient()
        else:
            # Resolve directory from parameter, env var, or project default
            target_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIR")
            if not target_dir:
                # Default path is backend/data/chroma relative to this file
                target_dir = (
                    Path(__file__).resolve().parent.parent.parent
                    / "data"
                    / "chroma"
                )
            self.persist_directory = Path(target_dir).resolve()
            # Ensure the local folder exists before pointing ChromaDB to it
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            # PersistentClient saves data to disk across restarts
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        # Get or create the collection with cosine distance metric for text embeddings
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # --- Embedding Model Management -----------------------------------------

    @property
    def embedding_model(self) -> Any:
        """
        Return the embedding model, lazily loading SentenceTransformer on first use.

        Lazy loading ensures that importing this module or creating a service
        instance does not block on downloading or initializing neural network weights
        until an embedding is actually computed.
        """
        if self._embedding_model is None:
            # Import sentence_transformers only when actually needed
            from sentence_transformers import SentenceTransformer

            # Load the local transformer model
            self._embedding_model = SentenceTransformer(self._model_name)
        return self._embedding_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text strings into vector embeddings.

        Parameters
        ----------
        texts : list[str]
            Text snippets to embed.

        Returns
        -------
        list[list[float]]
            A list of float vectors, one per input text snippet.
        """
        if not texts:
            return []

        # Use the embedding model to encode the texts
        raw_embeddings = self.embedding_model.encode(texts)

        # Convert numpy arrays or tensors into standard Python list of floats
        if hasattr(raw_embeddings, "tolist"):
            return raw_embeddings.tolist()

        # Fallback list conversion if already in list format
        return [list(vec) if not isinstance(vec, list) else vec for vec in raw_embeddings]

    # --- Validation Helpers -------------------------------------------------

    def _validate_vehicle_metadata(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, str | int | float | bool]:
        """
        Validate and sanitize vehicle metadata before storing in ChromaDB.

        ChromaDB metadata values must be primitive types (str, int, float, bool).
        Required fields: manufacturer (non-empty str), model (non-empty str), year (int).
        """
        if not isinstance(metadata, dict):
            raise ValueError(
                "vehicle_metadata must be a dictionary containing vehicle information."
            )

        # Validate manufacturer
        manufacturer = metadata.get("manufacturer")
        if not isinstance(manufacturer, str) or not manufacturer.strip():
            raise ValueError(
                "vehicle_metadata must contain a non-empty string for 'manufacturer'."
            )

        # Validate model
        model = metadata.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "vehicle_metadata must contain a non-empty string for 'model'."
            )

        # Validate year (in Python, bool is a subclass of int, so reject bool explicitly)
        year = metadata.get("year")
        if isinstance(year, bool) or not isinstance(year, int):
            raise ValueError(
                "vehicle_metadata must contain an integer for 'year'."
            )
        if year < 1900 or year > 2100:
            raise ValueError(
                f"vehicle_metadata 'year' must be between 1900 and 2100, got {year}."
            )

        # Clean base metadata
        cleaned: dict[str, str | int | float | bool] = {
            "manufacturer": manufacturer.strip(),
            "model": model.strip(),
            "year": int(year),
        }

        # Include additional primitive metadata fields if present
        for key, val in metadata.items():
            if key in ("manufacturer", "model", "year"):
                continue
            if isinstance(val, (str, int, float, bool)):
                cleaned[key] = val
            elif val is None:
                continue
            else:
                raise ValueError(
                    f"Invalid metadata value for key '{key}': ChromaDB only supports "
                    f"primitive types (str, int, float, bool), got {type(val).__name__}."
                )

        return cleaned

    def _validate_chunks(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """
        Validate that chunks match the TextChunk structure and have non-empty text.
        """
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list of TextChunk items.")

        validated: list[TextChunk] = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk at index {i} must be a dictionary.")

            source = chunk.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"Chunk at index {i} is missing a valid 'source' filename.")

            page_number = chunk.get("page_number")
            if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
                raise ValueError(
                    f"Chunk at index {i} must have a 1-based integer 'page_number' (>= 1)."
                )

            chunk_index = chunk.get("chunk_index")
            if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
                raise ValueError(
                    f"Chunk at index {i} must have a 0-based integer 'chunk_index' (>= 0)."
                )

            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Chunk at index {i} has empty or missing 'text'.")

            validated.append(chunk)

        return validated

    def _generate_chunk_id(
        self,
        vehicle_meta: dict[str, Any],
        source: str,
        page_number: int,
        chunk_index: int,
    ) -> str:
        """
        Generate a deterministic, unique string ID for a chunk.

        Format: {manufacturer}_{model}_{year}_{source}_p{page_number}_c{chunk_index}
        Example: toyota_camry_2020_camry_manual.pdf_p1_c0
        """
        make = str(vehicle_meta["manufacturer"]).lower().replace(" ", "_")
        model = str(vehicle_meta["model"]).lower().replace(" ", "_")
        year = str(vehicle_meta["year"])
        src = Path(source).name.lower().replace(" ", "_")
        return f"{make}_{model}_{year}_{src}_p{page_number}_c{chunk_index}"

    # --- Public Indexing Methods --------------------------------------------

    def index_chunks(
        self,
        chunks: list[TextChunk],
        vehicle_metadata: dict[str, Any],
    ) -> int:
        """
        Index a list of text chunks into ChromaDB with vehicle metadata.

        Parameters
        ----------
        chunks : list[TextChunk]
            List of chunks produced by chunker.chunk_pages().
        vehicle_metadata : dict[str, Any]
            Metadata describing the vehicle (manufacturer, model, year, etc.).

        Returns
        -------
        int
            The number of chunks successfully indexed. Returns 0 for empty input.
        """
        # Handle empty input cleanly without error
        if not chunks:
            return 0

        # Validate inputs before touching the database
        clean_meta = self._validate_vehicle_metadata(vehicle_metadata)
        valid_chunks = self._validate_chunks(chunks)

        # Extract text snippets to embed
        texts = [c["text"] for c in valid_chunks]

        # Generate embeddings using the configured embedding model
        embeddings = self.embed_texts(texts)

        # Prepare lists for ChromaDB batch insertion
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in valid_chunks:
            chunk_id = self._generate_chunk_id(
                clean_meta,
                chunk["source"],
                chunk["page_number"],
                chunk["chunk_index"],
            )
            ids.append(chunk_id)
            documents.append(chunk["text"])

            # Merge chunk positioning details with vehicle metadata
            meta = dict(clean_meta)
            meta["source"] = chunk["source"]
            meta["page_number"] = chunk["page_number"]
            meta["chunk_index"] = chunk["chunk_index"]
            metadatas.append(meta)

        # upsert replaces records if IDs already exist or inserts them if new.
        # This makes re-running the indexer idempotent without crashing on duplicate keys.
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(valid_chunks)

    # --- Query and Retrieval Methods ----------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedRecord]:
        """
        Query the vector index for chunks semantically similar to query_text.

        Parameters
        ----------
        query_text : str
            The search query (e.g. 'oil pressure warning light').
        n_results : int
            Maximum number of matching chunks to return (default 5).
        where : dict, optional
            ChromaDB filter dictionary to restrict results by metadata.
            Example: {'model': 'Camry'}

        Returns
        -------
        list[RetrievedRecord]
            List of matching records with their text, source, page, and metadata.
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string.")

        if isinstance(n_results, bool) or not isinstance(n_results, int) or n_results <= 0:
            raise ValueError("n_results must be a positive integer (>= 1).")

        # If the collection has no documents, return empty list immediately
        total_count = self.get_count()
        if total_count == 0:
            return []

        # Convert query text into an embedding vector
        query_embeddings = self.embed_texts([query_text])
        query_vector = query_embeddings[0]

        # ChromaDB cannot return more results than there are items in the collection
        limit = min(n_results, total_count)

        # Prepare query arguments
        query_args: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": limit,
        }
        if where:
            query_args["where"] = where

        # Execute search in ChromaDB
        results = self.collection.query(**query_args)

        # Unpack ChromaDB's nested lists into clean RetrievedRecord dicts
        records: list[RetrievedRecord] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return records

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i in range(len(ids)):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            doc = docs[i] if i < len(docs) else ""
            dist = distances[i] if i < len(distances) else None

            records.append(
                RetrievedRecord(
                    id=ids[i],
                    text=doc,
                    source=str(meta.get("source", "")),
                    page_number=int(meta.get("page_number", 0)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    manufacturer=str(meta.get("manufacturer", "")),
                    model=str(meta.get("model", "")),
                    year=int(meta.get("year", 0)),
                    metadata=meta,
                    distance=dist,
                )
            )

        return records

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[RetrievedRecord]:
        """
        Fetch stored records directly by ID or metadata filter without vector similarity.
        """
        get_kwargs: dict[str, Any] = {}
        if ids is not None:
            get_kwargs["ids"] = ids
        if where is not None:
            get_kwargs["where"] = where
        if limit is not None:
            get_kwargs["limit"] = limit

        results = self.collection.get(**get_kwargs)

        records: list[RetrievedRecord] = []
        if not results or not results.get("ids"):
            return records

        ids_list = results["ids"]
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []

        for i, chunk_id in enumerate(ids_list):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            doc = docs[i] if i < len(docs) else ""

            records.append(
                RetrievedRecord(
                    id=chunk_id,
                    text=doc,
                    source=str(meta.get("source", "")),
                    page_number=int(meta.get("page_number", 0)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    manufacturer=str(meta.get("manufacturer", "")),
                    model=str(meta.get("model", "")),
                    year=int(meta.get("year", 0)),
                    metadata=meta,
                    distance=None,
                )
            )

        return records

    def get_count(self) -> int:
        """Return the total number of records currently stored in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """
        Delete all stored records and re-initialize an empty collection.
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
