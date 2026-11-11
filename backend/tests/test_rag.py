"""
test_rag.py — Tests for SCRUM-10: PDF Extraction & Chunking (RAG Slice 1).

All PDFs used in these tests are generated in memory using PyMuPDF itself,
so no binary fixture files need to be committed to the repository.

Tested behaviours:
  Extraction:
    - A normal single-page PDF returns one ExtractedPage with correct text.
    - The page_number is 1-based (first page → 1, not 0).
    - The source field is the PDF filename, not the full path.
    - A multi-page PDF returns one entry per page in reading order.
    - An empty PDF (zero pages) returns an empty list without error.
    - A page with no selectable text returns an ExtractedPage with text="".
    - extract_pages() raises FileNotFoundError for a missing path.
    - extract_pages() raises ValueError for a file that is not a PDF.

  Chunking:
    - A text shorter than chunk_size yields a single chunk.
    - chunk_size and overlap produce the correct number of chunks.
    - Overlap characters are shared between consecutive chunks.
    - The chunk_index is 0-based and increases across the whole document.
    - Every chunk carries the correct source and page_number metadata.
    - Pages with no text produce no chunks (they are skipped).
    - chunk_size ≤ 0 raises ValueError.
    - overlap < 0 raises ValueError.
    - overlap ≥ chunk_size raises ValueError.
"""

import tempfile
from pathlib import Path

import pymupdf
import pytest

from app.rag.chunker import chunk_pages
from app.rag.extractor import extract_pages


# ===========================================================================
# Helpers — build minimal in-memory PDFs for testing
# ===========================================================================

def _make_pdf_bytes(pages_text: list[str]) -> bytes:
    """
    Create a valid PDF in memory containing one page per string in pages_text.

    Each string is written as a single line of text at a fixed position on
    its page.  This helper keeps tests self-contained and reproducible.

    Parameters
    ----------
    pages_text : list[str]
        One string per page.  Use "" or None to create a page with no text.

    Returns
    -------
    bytes
        Raw PDF bytes that can be written to a .pdf file.
    """
    doc = pymupdf.open()  # create an empty in-memory PDF

    for text in pages_text:
        # Add a blank A4-sized page to the document
        page = doc.new_page(width=595, height=842)

        if text:
            # Insert the text as a text block near the top of the page.
            # rect defines the bounding box: (x0, y0, x1, y1)
            rect = pymupdf.Rect(50, 50, 545, 792)
            page.insert_textbox(rect, text, fontsize=12)

    # tobytes() serialises the in-memory PDF without writing to disk
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _write_temp_pdf(pages_text: list[str], suffix: str = "test_manual.pdf") -> Path:
    """
    Write an in-memory PDF to a temporary file and return its Path.

    The file is not deleted automatically; tests manage cleanup via
    pytest tmp_path or explicit deletion where needed.
    """
    pdf_bytes = _make_pdf_bytes(pages_text)
    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{suffix}",
        delete=False,
    )
    tmp.write(pdf_bytes)
    tmp.close()
    return Path(tmp.name)


# ===========================================================================
# Extraction tests
# ===========================================================================

class TestExtractPages:
    """Tests for app.rag.extractor.extract_pages"""

    def test_single_page_returns_one_result(self):
        """A one-page PDF must produce exactly one ExtractedPage."""
        pdf_path = _write_temp_pdf(["Hello from page one."])
        result = extract_pages(pdf_path)

        assert len(result) == 1

    def test_text_content_is_preserved(self):
        """The text on the page must appear in the extracted result."""
        content = "Engine oil warning light: stop the vehicle safely."
        pdf_path = _write_temp_pdf([content])
        result = extract_pages(pdf_path)

        # We check 'in' rather than exact equality because PyMuPDF may add
        # newline characters when reflowing text boxes.
        assert content in result[0]["text"]

    def test_page_number_is_one_based(self):
        """The first page must have page_number == 1, not 0."""
        pdf_path = _write_temp_pdf(["First page text."])
        result = extract_pages(pdf_path)

        assert result[0]["page_number"] == 1

    def test_multi_page_page_numbers_are_sequential(self):
        """A 3-page PDF must return page_numbers 1, 2, 3 in order."""
        pdf_path = _write_temp_pdf(["Page A.", "Page B.", "Page C."])
        result = extract_pages(pdf_path)

        assert len(result) == 3
        assert [r["page_number"] for r in result] == [1, 2, 3]

    def test_source_is_filename_not_full_path(self):
        """The source field must be just the filename, not the full directory path."""
        pdf_path = _write_temp_pdf(["Some text."], suffix="my_manual.pdf")
        result = extract_pages(pdf_path)

        # source should be "..._my_manual.pdf", not contain directory separators
        assert "/" not in result[0]["source"]
        assert "\\" not in result[0]["source"]
        # and it must end with the expected suffix
        assert result[0]["source"].endswith("_my_manual.pdf")

    def test_source_matches_pdf_filename(self):
        """The source field must equal exactly the PDF's filename (with extension)."""
        pdf_path = _write_temp_pdf(["Text."])
        result = extract_pages(pdf_path)

        assert result[0]["source"] == pdf_path.name

    def test_empty_pdf_returns_empty_list(self, monkeypatch):
        """
        A PDF with zero pages must return an empty list without raising an error.

        PyMuPDF 1.28.2 cannot serialise a zero-page document to bytes
        (it raises "cannot save with zero pages"), so we cannot write one to
        disk.  Instead, we monkeypatch pymupdf.open inside the extractor module
        so it returns an in-memory zero-page Document directly.
        This still exercises the real iteration logic in extract_pages().
        """
        import app.rag.extractor as extractor_module

        # Create a zero-page document that lives only in memory.
        zero_page_doc = pymupdf.open()

        # We need a real file on disk so that extract_pages() passes its
        # file-existence check before calling pymupdf.open.
        tmp = tempfile.NamedTemporaryFile(suffix="_empty.pdf", delete=False)
        tmp.write(b"%PDF-1.4")  # minimal bytes — content doesn't matter here
        tmp.close()

        # Replace pymupdf.open *inside* the extractor module with a fake that
        # returns our zero-page document.  monkeypatch automatically restores
        # the real function when the test finishes.
        def fake_open(path):  # noqa: ANN001, ANN201
            return zero_page_doc

        monkeypatch.setattr(extractor_module.pymupdf, "open", fake_open)

        result = extract_pages(tmp.name)

        assert result == []

        # extract_pages() closes the document via 'with doc:', so we do NOT
        # call zero_page_doc.close() here — it would raise ValueError.

    def test_page_with_no_text_returns_empty_string(self):
        """A blank page (no text inserted) must produce an ExtractedPage with text==''."""
        # Pass "" so _make_pdf_bytes skips inserting text → purely blank page
        pdf_path = _write_temp_pdf([""])
        result = extract_pages(pdf_path)

        assert len(result) == 1
        assert result[0]["text"] == ""

    def test_missing_file_raises_file_not_found(self):
        """extract_pages must raise FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            extract_pages("/non/existent/path/manual.pdf")

    def test_non_pdf_file_raises_value_error(self):
        """extract_pages must raise ValueError when the file is not a valid PDF."""
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(b"this is not pdf content at all")
        tmp.close()

        with pytest.raises(ValueError):
            extract_pages(tmp.name)


# ===========================================================================
# Chunking tests
# ===========================================================================

class TestChunkPages:
    """Tests for app.rag.chunker.chunk_pages"""

    def _make_pages(self, texts: list[str], source: str = "test.pdf") -> list:
        """Build a minimal list of ExtractedPage dicts for chunking tests."""
        return [
            {"source": source, "page_number": i + 1, "text": text}
            for i, text in enumerate(texts)
        ]

    # --- Chunk count and content ------------------------------------------

    def test_text_shorter_than_chunk_size_yields_one_chunk(self):
        """Text that fits in a single chunk must produce exactly one chunk."""
        pages = self._make_pages(["Short text."])
        chunks = chunk_pages(pages, chunk_size=500, overlap=0)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."

    def test_correct_number_of_chunks_produced(self):
        """
        Verify the sliding-window chunk count.

        text = "ABCDEFGHIJ" (10 chars)
        chunk_size=4, overlap=1  →  step = 3
        starts: 0, 3, 6, 9  →  4 chunks
        """
        text = "ABCDEFGHIJ"   # 10 characters
        pages = self._make_pages([text])
        chunks = chunk_pages(pages, chunk_size=4, overlap=1)

        assert len(chunks) == 4

    def test_overlap_characters_are_shared(self):
        """
        Consecutive chunks must share exactly 'overlap' trailing/leading characters.

        text = "ABCDEFGHIJ" (10 chars), chunk_size=5, overlap=2
        step = 3
        Chunk 0: text[0:5]  = "ABCDE"
        Chunk 1: text[3:8]  = "DEFGH"
        Shared characters: "DE" (overlap=2)
        """
        text = "ABCDEFGHIJ"
        pages = self._make_pages([text])
        chunks = chunk_pages(pages, chunk_size=5, overlap=2)

        # chunk 0 ends with "DE"; chunk 1 starts with "DE"
        assert chunks[0]["text"][-2:] == chunks[1]["text"][:2]

    def test_chunk_index_is_zero_based_and_sequential(self):
        """chunk_index must start at 0 and increase by 1 for each chunk."""
        text = "A" * 30  # 30 characters
        pages = self._make_pages([text])
        chunks = chunk_pages(pages, chunk_size=10, overlap=0)

        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_index_continues_across_pages(self):
        """
        chunk_index must not reset to 0 at the start of each page.
        It must be global across the whole document.
        """
        # 10 chars per page → 1 chunk each with chunk_size=20, overlap=0
        pages = self._make_pages(["First page.", "Second page."])
        chunks = chunk_pages(pages, chunk_size=20, overlap=0)

        # Page 1 produces 1 chunk (chunk_index=0), page 2 continues (chunk_index=1)
        assert chunks[0]["chunk_index"] == 0
        assert chunks[1]["chunk_index"] == 1

    # --- Metadata preservation -------------------------------------------

    def test_source_is_preserved_on_every_chunk(self):
        """Every chunk must carry the source filename from its page."""
        pages = self._make_pages(["A" * 30], source="owners_manual.pdf")
        chunks = chunk_pages(pages, chunk_size=10, overlap=0)

        for chunk in chunks:
            assert chunk["source"] == "owners_manual.pdf"

    def test_page_number_is_preserved_on_every_chunk(self):
        """Every chunk must carry the page_number from its source page."""
        pages = self._make_pages(["A" * 30])
        chunks = chunk_pages(pages, chunk_size=10, overlap=0)

        # All chunks came from page 1
        for chunk in chunks:
            assert chunk["page_number"] == 1

    def test_page_number_correct_for_each_page(self):
        """Chunks from different pages must carry their respective page numbers."""
        pages = self._make_pages(["A" * 25, "B" * 25])
        chunks = chunk_pages(pages, chunk_size=10, overlap=0)

        page_1_chunks = [c for c in chunks if c["page_number"] == 1]
        page_2_chunks = [c for c in chunks if c["page_number"] == 2]

        # Both pages are 25 chars, step=10 → ceil(25/10) = 3 chunks each
        assert len(page_1_chunks) == 3
        assert len(page_2_chunks) == 3

    # --- Empty page handling ---------------------------------------------

    def test_empty_pages_produce_no_chunks(self):
        """Pages with no text must be skipped and produce no chunks."""
        pages = self._make_pages(["", ""])  # two blank pages
        chunks = chunk_pages(pages, chunk_size=100, overlap=0)

        assert chunks == []

    def test_mixed_empty_and_nonempty_pages(self):
        """Empty pages must be skipped; non-empty pages must still be chunked."""
        pages = self._make_pages(["", "Hello world.", ""])
        chunks = chunk_pages(pages, chunk_size=100, overlap=0)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello world."
        # The only chunk came from the second page (page_number=2)
        assert chunks[0]["page_number"] == 2

    # --- Input validation ------------------------------------------------

    def test_zero_chunk_size_raises_value_error(self):
        """chunk_size=0 must raise ValueError."""
        pages = self._make_pages(["Some text."])
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_pages(pages, chunk_size=0, overlap=0)

    def test_negative_chunk_size_raises_value_error(self):
        """Negative chunk_size must raise ValueError."""
        pages = self._make_pages(["Some text."])
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_pages(pages, chunk_size=-10, overlap=0)

    def test_negative_overlap_raises_value_error(self):
        """Negative overlap must raise ValueError."""
        pages = self._make_pages(["Some text."])
        with pytest.raises(ValueError, match="overlap"):
            chunk_pages(pages, chunk_size=100, overlap=-1)

    def test_overlap_equal_to_chunk_size_raises_value_error(self):
        """overlap equal to chunk_size must raise ValueError."""
        pages = self._make_pages(["Some text."])
        with pytest.raises(ValueError, match="overlap"):
            chunk_pages(pages, chunk_size=50, overlap=50)

    def test_overlap_greater_than_chunk_size_raises_value_error(self):
        """overlap larger than chunk_size must raise ValueError."""
        pages = self._make_pages(["Some text."])
        with pytest.raises(ValueError, match="overlap"):
            chunk_pages(pages, chunk_size=50, overlap=60)

    # --- End-to-end: extraction → chunking together ----------------------

    def test_extract_then_chunk_preserves_page_number(self):
        """
        End-to-end smoke test: extract from a real in-memory PDF,
        then chunk, and verify page_number is correctly propagated.
        """
        pdf_path = _write_temp_pdf(
            ["Warning lights indicate a problem. Stop the vehicle safely."],
        )
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, chunk_size=20, overlap=5)

        # All chunks must reference page 1 and the correct source filename
        for chunk in chunks:
            assert chunk["page_number"] == 1
            assert chunk["source"] == pdf_path.name

        # Text from the PDF must appear across the chunks
        full_reconstructed = "".join(c["text"] for c in chunks)
        # The original text must be a substring of the reconstructed (overlapping) text
        original_text = pages[0]["text"]
        assert original_text[: len(chunks[0]["text"])] == chunks[0]["text"]


# ===========================================================================
# Two-page integration tests — extraction then chunking across multiple pages
# ===========================================================================

class TestTwoPageIntegration:
    """
    End-to-end tests that build a real two-page in-memory PDF,
    run extract_pages() on it, and then run chunk_pages() on the result.

    These tests confirm that page numbers, source filenames, and text
    are all correct when both functions are used together in sequence,
    which is the normal usage path in the AutoLens RAG pipeline.
    """

    # Shared text content for both pages — intentionally distinct so we can
    # tell which page a chunk came from.
    PAGE_1_TEXT = "Oil pressure warning: pull over immediately and switch off the engine."
    PAGE_2_TEXT = "Battery charge warning: drive to a service centre as soon as possible."

    def _build_two_page_pdf(self) -> Path:
        """Write a two-page PDF to a temp file and return its Path."""
        return _write_temp_pdf(
            [self.PAGE_1_TEXT, self.PAGE_2_TEXT],
            suffix="two_page_manual.pdf",
        )

    def test_extraction_yields_two_pages(self):
        """
        extract_pages() must return exactly two ExtractedPage dicts,
        one for each page in the PDF.
        """
        pdf_path = self._build_two_page_pdf()
        pages = extract_pages(pdf_path)

        assert len(pages) == 2

    def test_extraction_page_numbers_are_one_and_two(self):
        """
        The first extracted page must have page_number=1 and the
        second must have page_number=2.  Neither may be 0.
        """
        pdf_path = self._build_two_page_pdf()
        pages = extract_pages(pdf_path)

        assert pages[0]["page_number"] == 1
        assert pages[1]["page_number"] == 2

    def test_extraction_text_is_page_local(self):
        """
        Text from page 1 must appear only in the page-1 result, and
        text from page 2 must appear only in the page-2 result.
        The chunker has not run yet; this tests extraction in isolation.
        """
        pdf_path = self._build_two_page_pdf()
        pages = extract_pages(pdf_path)

        # PyMuPDF may add a trailing newline, so we use 'in' not '=='.
        assert self.PAGE_1_TEXT in pages[0]["text"]
        assert self.PAGE_2_TEXT in pages[1]["text"]

        # Neither page should contain the other page's content.
        assert self.PAGE_2_TEXT not in pages[0]["text"]
        assert self.PAGE_1_TEXT not in pages[1]["text"]

    def test_extraction_source_is_same_filename_on_both_pages(self):
        """
        Both pages must share the same source filename, because they
        came from the same PDF file.
        """
        pdf_path = self._build_two_page_pdf()
        pages = extract_pages(pdf_path)

        assert pages[0]["source"] == pdf_path.name
        assert pages[1]["source"] == pdf_path.name

    def test_chunks_carry_correct_page_numbers(self):
        """
        After chunking, every chunk must carry the page_number of the
        page it came from.  Chunks from page 1 must say page_number=1;
        chunks from page 2 must say page_number=2.

        This confirms that the chunker never mixes up which page a
        chunk belongs to, even when the global chunk_index keeps rising.
        """
        pdf_path = self._build_two_page_pdf()
        pages = extract_pages(pdf_path)
        # Use a small chunk_size so we get several chunks per page.
        chunks = chunk_pages(pages, chunk_size=25, overlap=5)

        # Split the chunks back into two groups by page number.
        page_1_chunks = [c for c in chunks if c["page_number"] == 1]
        page_2_chunks = [c for c in chunks if c["page_number"] == 2]

        # Both pages have text, so both groups must be non-empty.
        assert len(page_1_chunks) > 0, "Expected at least one chunk from page 1"
        assert len(page_2_chunks) > 0, "Expected at least one chunk from page 2"

        # No chunk should have any other page number.
        for chunk in chunks:
            assert chunk["page_number"] in (1, 2)

    def test_chunks_source_is_preserved_on_all_chunks(self):
        """
        Every chunk — regardless of which page it came from — must
        carry the PDF filename as its source field.
        """
        pdf_path = self._build_two_page_pdf()
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, chunk_size=25, overlap=5)

        for chunk in chunks:
            assert chunk["source"] == pdf_path.name

    def test_chunk_index_is_globally_increasing_across_two_pages(self):
        """
        chunk_index must increase from 0 without ever resetting to 0
        when the chunker moves from page 1 to page 2.

        Example: if page 1 produces 4 chunks (indices 0-3), the first
        chunk from page 2 must be index 4, not 0.
        """
        pdf_path = self._build_two_page_pdf()
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, chunk_size=25, overlap=5)

        indices = [c["chunk_index"] for c in chunks]

        # Indices must start at 0, be unique, and be strictly increasing.
        assert indices[0] == 0
        assert indices == list(range(len(chunks)))

    def test_chunking_is_page_local_text_does_not_cross_page_boundary(self):
        """
        The chunker processes each page independently.  No single chunk
        should contain text from more than one page.

        We detect a cross-page leak by checking that no chunk contains
        both PAGE_1_TEXT fragments AND PAGE_2_TEXT fragments at the same
        time.  Because the pages use completely different words, any
        cross-page mix would be detectable.
        """
        pdf_path = self._build_two_page_pdf()
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, chunk_size=25, overlap=5)

        # Pick unique words that appear only on one page.
        page_1_marker = "Oil"      # only in PAGE_1_TEXT
        page_2_marker = "Battery"  # only in PAGE_2_TEXT

        for chunk in chunks:
            # A chunk must NOT contain markers from both pages simultaneously.
            has_p1 = page_1_marker in chunk["text"]
            has_p2 = page_2_marker in chunk["text"]
            assert not (has_p1 and has_p2), (
                f"Chunk {chunk['chunk_index']} contains text from both pages: "
                f"{chunk['text']!r}"
            )

    def test_two_page_pdf_extraction_to_chunking_pipeline(self, tmp_path: Path):
        """
        Focused end-to-end integration test for the PDF extraction -> text chunking pipeline.

        Verifies all key requirements on real, unmocked components:
        1. Text from both pages is extracted into ExtractedPage objects.
        2. Resulting chunks contain expected page-specific text.
        3. Configured 15-character overlap between consecutive chunks of the same page is verified.
        4. Beginning and end of each page's extracted text are preserved, and text reconstructs cleanly.
        5. Chunking never merges text across page boundaries (restarts at offset 0 for each page).
        6. Every chunk retains correct source identifier, original page number, and sequential index.
        7. Page 1 text is strictly isolated from Page 2 text, and vice versa.
        """
        # Step 1: Create a real two-page PDF file with distinct, identifiable text on each page.
        # Long text ensures multiple chunks are produced per page.
        p1_marker = "OIL PRESSURE WARNING SYSTEM"
        p1_details = "Immediately pull over to a safe area and stop the engine to prevent catastrophic damage."
        p1_text = f"{p1_marker}: Low engine oil pressure detected. {p1_details}"

        p2_marker = "BATTERY CHARGING FAULT"
        p2_details = "Turn off auxiliary electronics and drive immediately to the nearest authorized service center."
        p2_text = f"{p2_marker}: Alternator output is insufficient. {p2_details}"

        # Write the two-page PDF to disk using PyMuPDF and pytest's tmp_path fixture
        pdf_bytes = _make_pdf_bytes([p1_text, p2_text])
        test_pdf_path = tmp_path / "two_page_troubleshooting_guide.pdf"
        test_pdf_path.write_bytes(pdf_bytes)

        # Step 2: Pass through real extraction pipeline (Requirement: real extractor, no mocks)
        extracted_pages = extract_pages(test_pdf_path)

        # Verification: Text from both pages is extracted correctly
        assert len(extracted_pages) == 2, "Expected exactly two extracted pages"
        assert extracted_pages[0]["page_number"] == 1, "Page 1 page_number must be 1"
        assert extracted_pages[1]["page_number"] == 2, "Page 2 page_number must be 2"
        assert p1_marker in extracted_pages[0]["text"], "Page 1 must contain page 1 text"
        assert p2_marker in extracted_pages[1]["text"], "Page 2 must contain page 2 text"
        assert p2_marker not in extracted_pages[0]["text"], "Page 1 must not contain page 2 text"
        assert p1_marker not in extracted_pages[1]["text"], "Page 2 must not contain page 1 text"

        # Step 3: Pass through real chunking pipeline (Requirement: real chunker, no mocks)
        # Using chunk_size=60, overlap=15 produces multiple overlapping chunks per page
        overlap = 15
        chunk_size = 60
        chunks = chunk_pages(extracted_pages, chunk_size=chunk_size, overlap=overlap)

        # Separate chunks by their page_number metadata
        page_1_chunks = [c for c in chunks if c["page_number"] == 1]
        page_2_chunks = [c for c in chunks if c["page_number"] == 2]

        # Verification: The resulting chunks contain expected page-specific text
        assert len(page_1_chunks) >= 2, "Page 1 must produce multiple chunks"
        assert len(page_2_chunks) >= 2, "Page 2 must produce multiple chunks"
        assert len(chunks) == len(page_1_chunks) + len(page_2_chunks), (
            "Every chunk must belong to either page 1 or page 2"
        )

        # Verification: Preserves the beginning and end of each page's extracted text
        # First chunk on Page 1 must start at index 0 of Page 1's extracted text
        assert page_1_chunks[0]["text"] == extracted_pages[0]["text"][:chunk_size], (
            "Page 1 first chunk does not match beginning of extracted text"
        )
        # Last chunk on Page 1 must end with the final characters of Page 1's extracted text
        assert extracted_pages[0]["text"].endswith(page_1_chunks[-1]["text"]), (
            "Page 1 extracted text does not end with last chunk content"
        )

        # First chunk on Page 2 must start at index 0 of Page 2's extracted text
        assert page_2_chunks[0]["text"] == extracted_pages[1]["text"][:chunk_size], (
            "Page 2 first chunk does not match beginning of extracted text"
        )
        # Last chunk on Page 2 must end with the final characters of Page 2's extracted text
        assert extracted_pages[1]["text"].endswith(page_2_chunks[-1]["text"]), (
            "Page 2 extracted text does not end with last chunk content"
        )

        # Verification: Full text reconstruction accounting for overlap
        # In an overlapping chunker, chunk 0 provides the first `chunk_size` characters,
        # and each subsequent chunk k advances by `chunk_size - overlap` characters.
        # Slicing each subsequent chunk by `chunk[overlap:]` yields its new contribution.
        reconstructed_p1 = page_1_chunks[0]["text"] + "".join(c["text"][overlap:] for c in page_1_chunks[1:])
        assert reconstructed_p1 == extracted_pages[0]["text"], (
            "Reconstructed Page 1 text from overlapping chunks does not match extracted page text"
        )

        reconstructed_p2 = page_2_chunks[0]["text"] + "".join(c["text"][overlap:] for c in page_2_chunks[1:])
        assert reconstructed_p2 == extracted_pages[1]["text"], (
            "Reconstructed Page 2 text from overlapping chunks does not match extracted page text"
        )

        # Verification: Explicit check of the configured 15-character overlap between consecutive chunks
        # The last 15 characters of chunk i must exactly match the first 15 characters of chunk i+1
        for i in range(len(page_1_chunks) - 1):
            assert page_1_chunks[i]["text"][-overlap:] == page_1_chunks[i + 1]["text"][:overlap], (
                f"Page 1 chunks {i} and {i + 1} do not share the expected {overlap}-character overlap"
            )
        for i in range(len(page_2_chunks) - 1):
            assert page_2_chunks[i]["text"][-overlap:] == page_2_chunks[i + 1]["text"][:overlap], (
                f"Page 2 chunks {i} and {i + 1} do not share the expected {overlap}-character overlap"
            )

        # Verification: Chunking never merges text across page boundaries
        # 1. The transition between the last chunk of Page 1 and the first chunk of Page 2 does NOT share overlap
        assert page_1_chunks[-1]["text"][-overlap:] != page_2_chunks[0]["text"][:overlap], (
            "Page boundary chunks must not share overlap characters"
        )
        # 2. No single chunk across the entire document contains text spanning both pages
        for chunk in chunks:
            has_p1_marker = "OIL" in chunk["text"].upper()
            has_p2_marker = "BATTERY" in chunk["text"].upper()
            assert not (has_p1_marker and has_p2_marker), (
                f"Chunk {chunk['chunk_index']} illegally contains text from both page 1 and page 2: {chunk['text']!r}"
            )

        # Verification: Every chunk retains correct source identifier and original page number
        for idx, chunk in enumerate(chunks):
            # Source must match the PDF filename without directory path
            assert chunk["source"] == test_pdf_path.name, (
                f"Chunk {idx} source ({chunk['source']}) must equal PDF filename ({test_pdf_path.name})"
            )
            # Page number must match original PDF page (1 or 2)
            assert chunk["page_number"] in (1, 2), f"Chunk {idx} has invalid page number {chunk['page_number']}"
            # Global chunk index must be sequential starting at 0
            assert chunk["chunk_index"] == idx, f"Chunk index {chunk['chunk_index']} does not match expected {idx}"
            # Chunk length must not exceed chunk_size
            assert len(chunk["text"]) <= chunk_size, (
                f"Chunk {idx} exceeds chunk_size (length {len(chunk['text'])})"
            )

        # Verification: Page 1 text is not incorrectly attributed to page 2, or vice versa
        # Verify page 1 chunks strictly contain page 1 text and no page 2 identifiers
        for chunk in page_1_chunks:
            assert chunk["text"] in extracted_pages[0]["text"], (
                "Page 1 chunk text must originate from Page 1 extracted text"
            )
            assert "BATTERY" not in chunk["text"].upper(), "Page 1 chunk must not contain page 2 keyword BATTERY"
            assert "ALTERNATOR" not in chunk["text"].upper(), "Page 1 chunk must not contain page 2 keyword ALTERNATOR"

        # Verify page 2 chunks strictly contain page 2 text and no page 1 identifiers
        for chunk in page_2_chunks:
            assert chunk["text"] in extracted_pages[1]["text"], (
                "Page 2 chunk text must originate from Page 2 extracted text"
            )
            assert "OIL" not in chunk["text"].upper(), "Page 2 chunk must not contain page 1 keyword OIL"
            assert "DAMAGE" not in chunk["text"].upper(), "Page 2 chunk must not contain page 1 keyword DAMAGE"


