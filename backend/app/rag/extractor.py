"""
extractor.py — PDF text extraction for AutoLens RAG pipeline.

This module reads a vehicle owner's manual PDF page by page and returns
the text from each page together with its page number and the name of the
source file.  No AI models or network calls are involved.

Usage example:
    from app.rag.extractor import extract_pages

    pages = extract_pages("data/sample_manual.pdf")
    for page in pages:
        print(page["page_number"], page["text"][:80])
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pymupdf  # PyMuPDF — already installed and declared in requirements.txt


# ---------------------------------------------------------------------------
# Data structure for a single extracted page
# ---------------------------------------------------------------------------

class ExtractedPage(TypedDict):
    """
    Holds the text and metadata for one PDF page.

    Attributes
    ----------
    source : str
        The filename of the PDF (e.g. "toyota_camry_2020.pdf").
        Used later to identify which manual a chunk came from.
    page_number : int
        1-based page number, matching what is printed in the PDF viewer.
        Page 1 in the PDF becomes page_number 1 here (not 0).
    text : str
        The extracted plain text from that page.
        Empty string ("") when the page contains no machine-readable text
        (e.g. a purely scanned/image page).
    """

    source: str
    page_number: int
    text: str


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: str | Path) -> list[ExtractedPage]:
    """
    Extract text from every page of a PDF, one page at a time.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file on disk.

    Returns
    -------
    list[ExtractedPage]
        One dictionary per page, in reading order, containing:
        - "source"      : filename of the PDF (no directory prefix)
        - "page_number" : 1-based page number (int)
        - "text"        : extracted text (may be empty string "")

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the path points to a file that is not a valid PDF.

    Notes
    -----
    - An empty PDF (0 pages) returns an empty list; this is not an error. 
    - A page with no selectable text (e.g. a scanned image without OCR)
      returns an ExtractedPage with text="".  The caller can decide whether
      to skip or flag such pages.
    - The PDF is always closed properly, even if an error occurs, because
      we open it with a 'with' statement.
    """

    # Convert to a Path object so we can use .name and .exists() cleanly
    pdf_path = Path(pdf_path)

    # Check that the file actually exists before trying to open it
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Use just the filename (e.g. "manual.pdf") as the source identifier,
    # stripping any directory path information.
    source_name = pdf_path.name

    # Open the PDF using a 'with' statement so it is always closed safely,
    # even if something goes wrong inside the block.
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        # pymupdf raises a generic Exception for files that are not valid PDFs.
        raise ValueError(f"Could not open '{pdf_path}' as a PDF: {exc}") from exc

    extracted: list[ExtractedPage] = []

    with doc:  # 'with doc' ensures doc.close() is called automatically
        # doc is a sequence of pages; we iterate using enumerate for the index
        for index, page in enumerate(doc):
            # pymupdf uses 0-based page indices internally.
            # We add 1 so that the first page becomes page_number 1, matching
            # the page number a reader would see in a PDF viewer.
            page_number = index + 1

            # get_text() returns all selectable text on the page as a string.
            # It returns "" for pages that are pure images (scanned PDFs).
            raw_text = page.get_text()

            # Strip leading and trailing whitespace; keeps internal spacing intact.
            cleaned_text = raw_text.strip()

            extracted.append(
                ExtractedPage(
                    source=source_name,
                    page_number=page_number,
                    text=cleaned_text,
                )
            )

    return extracted
