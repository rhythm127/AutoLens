from fastapi.testclient import TestClient
from app.main import app

# Initialize FastAPI test client
client = TestClient(app)


def test_health_check_returns_ok():
    # Make a GET request to the /health endpoint
    response = client.get("/health")

    # Assert HTTP status code is 200 OK
    assert response.status_code == 200

    # Assert expected payload is returned
    assert response.json() == {
        "status": "ok",
        "app": "AutoLens",
    }
