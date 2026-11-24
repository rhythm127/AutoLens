from app.db.database import Database
from test_chat import VEHICLE


def test_transcript_persists_across_requests_and_database_instances(api, manual):
    client, database = api
    first = client.post("/chat", json={**VEHICLE, "message": "My engine light is steady"}).json()
    second = client.post("/chat", json={"session_id": first["session_id"], "message": "It is now flashing"}).json()
    session = client.get(f'/history/sessions/{first["session_id"]}').json()
    assert [m["role"] for m in session["messages"]] == ["user", "assistant", "user", "assistant"]
    assert session["messages"][-1]["citations"] == second["citations"]
    assert session["messages"][-1]["severity"] == second["severity"]
    assert session["severity"] == "HIGH"
    assert Database(database.path).session(first["session_id"]) == session
    assert len(client.get("/history/sessions", params={"vehicle_id": session["vehicle_id"]}).json()["sessions"]) == 1
    assert client.get("/history/sessions", params={"vehicle_id": "another"}).json()["sessions"] == []
    assert client.get("/vehicles").json()["vehicles"][0]["id"] == session["vehicle_id"]


def test_unknown_ids_and_vehicle_switch_rejected(api, manual):
    client, database = api
    assert client.get("/history/sessions/missing").status_code == 404
    assert client.post("/chat", json={"session_id": "missing", "message": "hello"}).status_code == 404
    assert client.post("/chat", json={"vehicle_id": "missing", "message": "hello"}).status_code == 404
    first = client.post("/chat", json={**VEHICLE, "message": "engine light"}).json()
    other = database.save_vehicle("Toyota", "Corolla", 2020)
    response = client.post("/chat", json={"session_id": first["session_id"], "vehicle_id": other["id"], "message": "hello"})
    assert response.status_code == 409
    response = client.post("/chat", json={**VEHICLE, "model": "other", "session_id": first["session_id"], "message": "hello"})
    assert response.status_code == 409


def test_vehicle_upsert_preserves_manual(api):
    client, database = api
    first = database.save_vehicle(**VEHICLE, manual_filename="manual.pdf")
    assert client.post("/vehicles", json=VEHICLE).status_code == 200
    assert database.vehicles() == [first]


def test_cors_preflight(api):
    for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
        response = api[0].options("/chat", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                                   "Access-Control-Request-Headers": "content-type"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin


def test_manual_ingestion_persists_vehicle_and_source(api, monkeypatch):
    import fitz
    import app.main as main
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Check engine warning: consult an authorized dealer.")
    content = pdf.tobytes()
    pdf.close()
    indexed = []

    def index(chunks, metadata):
        indexed.extend(chunks)
        return len(chunks)

    monkeypatch.setattr(main.vector_service, "index_chunks", index)
    response = api[0].post("/rag/ingest", data=VEHICLE, files={"file": ("manual.pdf", content, "application/pdf")})
    assert response.status_code == 200
    result = response.json()
    assert result["pages_extracted"] == 1
    assert result["chunks_indexed"] == len(indexed) > 0
    assert indexed[0]["source"] == "manual.pdf"
    assert api[1].vehicle(result["vehicle_id"])["manual_filename"] == "manual.pdf"
