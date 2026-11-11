import uuid
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.indexer import VectorIndexService
from tests.test_indexer import MockEmbeddingModel


def test_hybrid_retriever_combines_dense_and_bm25_exact_code():
    """Verify HybridRetriever ranks exact OBD-II fault codes at the top via BM25 + RRF."""
    service = VectorIndexService(
        in_memory=True,
        collection_name=f"test_hybrid_{uuid.uuid4().hex}",
        embedding_model=MockEmbeddingModel(dimension=4),
    )

    chunks = [
        {
            "source": "camry_manual.pdf",
            "page_number": 10,
            "chunk_index": 0,
            "text": "General engine maintenance and checking the dipstick oil level.",
        },
        {
            "source": "camry_manual.pdf",
            "page_number": 45,
            "chunk_index": 1,
            "text": "Fault Code P0420: Catalyst System Efficiency Below Threshold (Bank 1).",
        },
    ]
    vehicle = {"manufacturer": "Toyota", "model": "Camry", "year": 2020}
    service.index_chunks(chunks, vehicle)

    retriever = HybridRetriever(vector_service=service)
    results = retriever.search("P0420", n_results=2, where={"manufacturer": "Toyota"})

    assert len(results) == 2
    # Page 45 with exact code P0420 must be ranked #1
    assert results[0]["page_number"] == 45
    assert "P0420" in results[0]["text"]
    assert "rrf_score" in results[0]["metadata"]