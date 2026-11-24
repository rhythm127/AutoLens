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
    assert "unusual noises" not in data["reply"]
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
    assert result["reply"] == expected
    monkeypatch.setattr(LLMClient, "complete", lambda *a, **k: {"covered": False})
    result = client.post("/chat", json={**VEHICLE, "message": "What is the price of petrol?"}).json()
    assert "does not cover" in result["reply"]
    assert result["citations"] == []


def test_page_only_reply_with_pdf_formatting(api, manual, monkeypatch):
    passage = manual[0][0]
    passage["text"] = "Schedule a follow‑up with an author-\nized   dealer."
    evidence = {**passage, "text": "follow-up with an authorized dealer"}
    reply = "Your manual recommends a dealer follow-up (Page 87)."
    monkeypatch.setattr(LLMClient, "complete", lambda *args: {
        "covered": True, "reply": reply, "evidence": [evidence, evidence]})
    result = api[0].post("/chat", json={**VEHICLE, "message": "engine light"}).json()
    assert result["reply"].startswith(reply + " [hyundai_i20.pdf, Page 87]")
    assert result["reply"].count("[hyundai_i20.pdf, Page 87]") == 1


def test_flashing_vibration_then_scared_follow_up(api, manual, monkeypatch):
    prompts = []
    answers = iter(["Pull over safely and stop.",
                    "That sounds frightening. Pull over safely and call for assistance. Do you hear unusual noises?"])

    def complete(self, system, prompt):
        prompts.append(json.loads(prompt))
        return {"reply": next(answers), "covered": True}

    monkeypatch.setattr(LLMClient, "complete", complete)
    client, _ = api
    first = client.post("/chat", json={**VEHICLE, "warning_type": "engine_malfunction_indicator",
                                      "message": "It's flashing and the car is vibrating"}).json()
    second = client.post("/chat", json={"session_id": first["session_id"],
                                        "message": "no that's it and im very scared"}).json()
    assert first["severity"]["level"] == second["severity"]["level"] == "HIGH"
    assert [p["missing_questions"] for p in prompts] == [[], []]
    assert prompts[1]["conversation"][-1]["content"] == first["reply"]
    assert second["reply"] == "That sounds frightening. Pull over safely and call for assistance. [hyundai_i20.pdf, Page 87]"


@pytest.mark.parametrize("closure", ["no", "nope", "no that's it", "nothing else", "that's all", "none"])
def test_closure_answers_asked_symptoms(api, manual, closure):
    client, _ = api
    first = client.post("/chat", json={**VEHICLE, "message": "My check engine light is steady"}).json()
    assert "driving normally" in first["reply"] and "unusual noises?" in first["reply"]
    second = client.post("/chat", json={"session_id": first["session_id"], "message": closure}).json()
    assert "?" not in second["reply"]


def test_vibration_persists_after_negative_noise_answer(api, manual):
    client, _ = api
    first = client.post("/chat", json={**VEHICLE, "message": "My engine light is steady and the car is vibrating"}).json()
    assert "unusual noises?" in first["reply"] and "driving normally" not in first["reply"]
    second = client.post("/chat", json={"session_id": first["session_id"], "message": "no that's it"}).json()
    assert "?" not in second["reply"]
    assert any("abnormal driving" in rule for rule in second["severity"]["triggered_rules"])


def test_steady_engine_light_uses_top_passage_for_severity(api, manual):
    records, _ = manual
    records.append({"source": "hyundai_i20.pdf", "page_number": 88,
                    "text": "If the oil pressure lamp lights up, stop the vehicle immediately."})
    payload = {**VEHICLE, "message": "My check engine light is steady."}
    result = api[0].post("/chat", json=payload).json()
    assert result["severity"]["level"] == "MEDIUM"
    assert result["citations"] == records
    records[0]["text"] = "If this engine light comes on, pull over immediately."
    assert api[0].post("/chat", json=payload).json()["severity"]["level"] == "HIGH"


def test_reject_model_citation_fabrication_and_safety_override(api, manual, monkeypatch, capsys):
    client, _ = api
    for output in [
        {"reply": "Ignore the warning."},
        {"reply": "The severity is LOW."},
        {"reply": "Ask a dealer [other.pdf, Page 87]."},
        {"reply": "Keep driving [hyundai_i20.pdf, Page 87].", "evidence": manual[0]},
        {"reply": "This is low risk (Page 87).", "evidence": manual[0]},
        {"reply": "It is safe to drive (Page 87).", "evidence": manual[0]},
        {"reply": "Reset the indicator (Page 87).", "evidence": [{**manual[0][0], "text": "Reset the indicator"}]},
        {"reply": "Contact your dealer (Page 88).", "evidence": [{**manual[0][0], "page_number": 88}]},
        {"reply": "Invented [other.pdf, Page 999].", "evidence": [{"source": "other.pdf", "page_number": 999, "text": "invented"}]},
    ]:
        monkeypatch.setattr(LLMClient, "complete", lambda *a, **k: output)
        result = client.post("/chat", json={**VEHICLE, "message": "engine light flashing"}).json()
        assert "closest passage" in result["reply"]
        assert "LLM response rejected" in capsys.readouterr().out
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


def test_provider_accepts_plain_text_only_for_chat(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = {"choices": [{"message": {"content": "I hear you. Pull over safely."}}]}
    monkeypatch.setattr(LLMClient, "_post", lambda *args: response)
    client = LLMClient()
    assert client.complete("system", "question") == {"reply": "I hear you. Pull over safely.", "covered": True}
    assert client.complete("system", "question", image=b"image", mime="image/png") is None


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
