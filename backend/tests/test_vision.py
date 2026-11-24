import pytest

from app.main import app, get_vision_analyzer
from app.vision.analyzer import VisionAnalyzer

IMAGE = b"\x89PNG\r\n\x1a\nimage-test-fixture"


@pytest.mark.parametrize("raw,status", [
    ('```json\n{"warning":"engine_malfunction_indicator","confidence":0.94,"description":"Amber engine symbol"}\n```', "detected"),
    ({"warning": "tpms_warning", "confidence": 0.65, "description": "Tyre warning"}, "detected"),
    ({"warning": "tpms_warning", "confidence": 0.64, "description": "Unclear symbol"}, "low_confidence"),
    ({"warning": "invalid", "confidence": 0.99, "description": "Unknown"}, "low_confidence"),
    ({"warning": [], "confidence": 0.99, "description": "Unknown"}, "low_confidence"),
    ({"warning": "tpms_warning", "confidence": float("nan"), "description": "Unknown"}, "low_confidence"),
    ({"warning": "tpms_warning", "confidence": True, "description": "Unknown"}, "low_confidence"),
    ({"warning": "tpms_warning", "confidence": 1.1, "description": "Unknown"}, "low_confidence"),
    ({"warning": "tpms_warning", "confidence": 0.99, "description": "Unclear", "unclear": True}, "low_confidence"),
    ("not JSON", "low_confidence"), ("[]", "low_confidence"), (None, "low_confidence"),
])
def test_vision_validation_endpoint(api, raw, status):
    class MockLLM:
        def complete(self, *args, **kwargs):
            return raw

    app.dependency_overrides[get_vision_analyzer] = lambda: VisionAnalyzer(MockLLM())
    response = api[0].post("/vision/analyze", files={"file": ("dash.png", IMAGE, "image/png")})
    assert response.status_code == 200
    assert response.json()["status"] == status
    if status == "low_confidence":
        assert response.json()["warning"] is None
        assert "retake" in response.json()["prompt"]


def test_image_upload_validation(api):
    client, _ = api
    assert client.post("/vision/analyze", files={"file": ("file.txt", b"text", "text/plain")}).status_code == 415
    assert client.post("/vision/analyze", files={"file": ("fake.png", b"text", "image/png")}).status_code == 400
    assert client.post("/vision/analyze", files={"file": ("large.png", IMAGE + b"x" * (10 * 1024 * 1024), "image/png")}).status_code == 413


def test_offline_vision_never_invents_warning(api):
    response = api[0].post("/vision/analyze", files={"file": ("dash.png", IMAGE, "image/png")})
    assert response.json()["status"] == "low_confidence"
