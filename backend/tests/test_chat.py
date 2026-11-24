import json
from urllib.error import URLError

import pytest

from app.chat.llm_client import LLMClient

VEHICLE = {"manufacturer": "Hyundai", "model": "i20", "year": 2024}


def test_grounded_offline_chat_with_severity(api, manual):
    client, _ = api
    response = client.post("/chat", json={**VEHICLE, "message": "My engine light is blinking and the car is shaking."})
    assert response.status_code == 200
    data = response.json()
    assert data["severity"]["level"] == "HIGH"
    assert data["severity"]["recommended_action"] in data["reply"]
    assert "hyundai_i20.pdf, Page 87" in data["reply"]
    assert data["citations"] == manual[0]
    assert manual[0][0]["text"] in data["reply"]
    assert "unusual noises" in data["reply"]
    assert "steady or flashing?" not in data["reply"]
    assert manual[1][0]["where"] == {"$and": [{k: v} for k, v in VEHICLE.items()]}


def test_follow_up_retains_vehicle_and_symptoms(api, manual):
    client, _ = api
    first = client.post("/chat", json={**VEHICLE, "message": "My engine light is flashing"}).json()
    second = client.post("/chat", json={"session_id": first["session_id"], "message": "No unusual noises, driving normally"}).json()
    assert second["severity"]["level"] == "HIGH"
    assert "Have you noticed" not in second["reply"]
    assert "Is the light" not in second["reply"]
    assert manual[1][0]["where"] == manual[1][1]["where"]


def test_missing_manual_and_unscoped_chat(api, manual):
    client, _ = api
    result = client.post("/chat", json={"message": "What is this warning?"}).json()
    assert "does not cover" in result["reply"]
    assert result["citations"] == []
    assert manual[1] == []
    manual[0].clear()
    result = client.post("/chat", json={**VEHICLE, "message": "engine light"}).json()
    assert "does not cover" in result["reply"]


def test_live_provider_reply_and_unsupported_question(api, manual, monkeypatch):
    client, _ = api
    expected = "Your manual says to contact an authorized dealer [hyundai_i20.pdf, Page 87]."
    monkeypatch.setattr(LLMClient, "complete", lambda *a, **k: {
        "covered": True, "reply": expected, "evidence": manual[0]})
    result = client.post("/chat", json={**VEHICLE, "message": "engine light"}).json()
    assert result["reply"].startswith(expected)
    monkeypatch.setattr(LLMClient, "complete", lambda *a, **k: {"covered": False})
    result = client.post("/chat", json={**VEHICLE, "message": "What is the price of petrol?"}).json()
    assert "does not cover" in result["reply"]
    assert result["citations"] == []


def test_reject_model_citation_fabrication_and_safety_override(api, manual, monkeypatch):
    client, _ = api
    for output in [
        {"reply": "Keep driving [hyundai_i20.pdf, Page 87].", "evidence": manual[0]},
        {"reply": "Invented [other.pdf, Page 999].", "evidence": [{"source": "other.pdf", "page_number": 999, "text": "invented"}]},
    ]:
        monkeypatch.setattr(LLMClient, "complete", lambda *a, **k: output)
        result = client.post("/chat", json={**VEHICLE, "message": "engine light flashing"}).json()
        assert "closest passage" in result["reply"]
        assert result["severity"]["level"] == "HIGH"


@pytest.mark.parametrize("payload", [{"message": "   "}, {"message": "hello", "warning_type": "made_up"},
                                    {"message": "hello", "manufacturer": "Hyundai"}])
def test_chat_validation(api, payload):
    assert api[0].post("/chat", json=payload).status_code == 422


def test_critical_user_warning_overrides_informational_vision_result(api, manual):
    result = api[0].post("/chat", json={**VEHICLE, "message": "My oil pressure warning is on",
                                        "warning_type": "low_fuel_warning"}).json()
    assert result["severity"]["level"] == "HIGH"


def test_provider_offline_fallback(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    calls = []

    def unavailable(self, url, body, **kwargs):
        calls.append(url)
        raise URLError("offline")

    monkeypatch.setattr(LLMClient, "_post", unavailable)
    assert LLMClient().complete("system", "question") is None
    assert calls == ["http://localhost:11434/api/chat"]


def test_provider_falls_back_to_ollama(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    calls = []

    def post(self, url, body, headers=None, **kwargs):
        calls.append((url, body))
        if "openrouter" in url:
            raise URLError("unavailable")
        return {"message": {"content": json.dumps({"covered": False})}}

    monkeypatch.setattr(LLMClient, "_post", post)
    assert LLMClient().complete("system", "question") == {"covered": False}
    assert len(calls) == 2


def test_gemini_image_payload(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def post(self, url, body, headers=None, **kwargs):
        calls.append(body)
        return {"candidates": [{"content": {"parts": [{"text": '{"warning":null}'}]}}]}

    monkeypatch.setattr(LLMClient, "_post", post)
    assert LLMClient().complete("system", "question", image=b"image", mime="image/png") == {"warning": None}
    assert calls[0]["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/png"
