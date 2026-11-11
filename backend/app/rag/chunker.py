"""
chunker.py — Text chunking for AutoLens RAG pipeline.

This module takes the list of extracted pages produced by extractor.py and
splits each page's text into smaller, overlapping chunks.  Keeping chunks
small lets the later retrieval step find the most relevant paragraph rather
than returning an entire page.  Overlap between consecutive chunks prevents
a sentence that straddles a chunk boundary from being split in a way that
loses its meaning.

No AI models, network calls, or new dependencies are required.

Usage example:
    from app.rag.extractor import extract_pages
    from app.rag.chunker import chunk_pages

    pages = extract_pages("data/sample_manual.pdf")
    chunks = chunk_pages(pages, chunk_size=400, overlap=80)
    for c in chunks:
        print(c["chunk_index"], c["page_number"], c["text"][:60])
"""

from __future__ import annotations

from typing import TypedDict

from app.rag.extractor import ExtractedPage


# ---------------------------------------------------------------------------
# Data structure for a single text chunk
# ---------------------------------------------------------------------------

class TextChunk(TypedDict):
    """
    Holds one text chunk and its metadata.

    Attributes
    ----------
    source : str
        Filename of the PDF this chunk came from (e.g. "manual.pdf").
    page_number : int
        1-based page number where this chunk's text begins.
    chunk_index : int
        0-based position of this chunk within the entire document.
        The first chunk of the whole document is chunk_index 0.
    text : str
        The actual text content of this chunk.
    """

    source: str
    page_number: int
    chunk_index: int
    text: str


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[ExtractedPage],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[TextChunk]:
    """
    Split a list of extracted pages into overlapping text chunks.

    How chunking works
    ------------------
    Imagine a single page's text as a long ribbon of characters.
    We slide a "window" of width ``chunk_size`` along that ribbon.
    After cutting each chunk, we step the window forward by
    ``chunk_size - overlap`` characters (not by the full ``chunk_size``).
    This means consecutive chunks share ``overlap`` characters, so a
    sentence near the boundary of one chunk also appears at the start
    of the next, preventing information from being cut in half.

    Simple example (chunk_size=10, overlap=3):
        Text: "ABCDEFGHIJKLMNOPQRST"
        Step = 10 - 3 = 7 characters

        Chunk 0: "ABCDEFGHIJ"   (characters 0–9)
        Chunk 1: "HIJKLMNOPQ"   (characters 7–16)
        Chunk 2: "OPQRST"       (characters 14–19, shorter because the text ended)

    Parameters
    ----------
    pages : list[ExtractedPage]
        Output from extractor.extract_pages().
    chunk_size : int
        Maximum number of characters in one chunk. Must be positive (≥ 1).
    overlap : int
        Number of characters that consecutive chunks share.
        Must be non-negative (≥ 0) and strictly less than chunk_size.

    Returns
    -------
    list[TextChunk]
        All chunks from all pages, in document order.
        Pages with empty text produce no chunks.

    Raises
    ------
    ValueError
        If chunk_size is not a positive integer (≤ 0).
        If overlap is negative or is greater than or equal to chunk_size.
    """

    # --- Input validation -----------------------------------------------
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be a positive integer, got {chunk_size}."
        )

    if overlap < 0:
        raise ValueError(
            f"overlap must be non-negative, got {overlap}."
        )

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size}). "
            "If overlap equals or exceeds chunk_size, the window never advances "
            "and chunking would loop forever."
        )

    # --- Chunking logic --------------------------------------------------

    all_chunks: list[TextChunk] = []

    # chunk_index counts chunks across the entire document (not per page)
    global_chunk_index = 0

    # How many characters to advance the window on each step.
    # This is always at least 1 because we validated overlap < chunk_size.
    step = chunk_size - overlap

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        source = page["source"]

        # Skip pages that have no text (e.g. blank cover pages, scanned images).
        # An empty page produces zero chunks, which is correct behaviour.
        if not text:
            continue

        # Slide the window from position 0 to the end of the text.
        start = 0
        while start < len(text):
            # The end of this chunk — Python slicing handles the case where
            # start + chunk_size goes beyond len(text) by stopping at the end.
            end = start + chunk_size

            # Extract this chunk's text
            chunk_text = text[start:end]

            all_chunks.append(
                TextChunk(
                    source=source,
                    page_number=page_number,
                    chunk_index=global_chunk_index,
                    text=chunk_text,
                )
            )

            global_chunk_index += 1
            start += step  # advance the window by (chunk_size - overlap)

    return all_chunks
