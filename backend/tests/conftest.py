import pytest
from fastapi.testclient import TestClient

from app.main import app, get_database
from app.db.database import Database
from app.chat.llm_client import LLMClient


@pytest.fixture
def api(tmp_path, monkeypatch):
    database = Database(tmp_path / "history.sqlite3")
    app.dependency_overrides[get_database] = lambda: database
    monkeypatch.setattr(LLMClient, "complete", lambda *args, **kwargs: None)
    with TestClient(app) as client:
        yield client, database
    app.dependency_overrides.clear()


@pytest.fixture
def manual(monkeypatch):
    import app.main as main
    records = [{"source": "hyundai_i20.pdf", "page_number": 87,
                "text": "If the malfunction indicator flashes, contact an authorized dealer."}]
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return records

    monkeypatch.setattr(main.hybrid_retriever, "search", search)
    return records, calls
