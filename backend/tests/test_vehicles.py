from fastapi.testclient import TestClient
from app.main import app

# Initialize FastAPI test client
client = TestClient(app)


def test_create_vehicle_valid():
    # Valid payload with leading and trailing spaces that should be stripped
    payload = {
        "manufacturer": "  Toyota  ",
        "model": "  Camry  ",
        "year": 2020,
    }
    response = client.post("/vehicles", json=payload)

    # Assert HTTP status code is 200 OK
    assert response.status_code == 200

    # Assert that spaces are stripped and echoed back properly
    assert response.json() == {
        "status": "received",
        "vehicle": {
            "manufacturer": "Toyota",
            "model": "Camry",
            "year": 2020,
        },
    }


def test_create_vehicle_year_1950_returns_422():
    # Year 1950 is below the allowed minimum of 1980
    payload = {
        "manufacturer": "Ford",
        "model": "Mustang",
        "year": 1950,
    }
    response = client.post("/vehicles", json=payload)

    # Assert HTTP status code is 422 Unprocessable Entity
    assert response.status_code == 422


def test_create_vehicle_missing_field_returns_422():
    # Payload is missing required 'model' field
    payload = {
        "manufacturer": "Honda",
        "year": 2022,
    }
    response = client.post("/vehicles", json=payload)

    # Assert HTTP status code is 422 Unprocessable Entity
    assert response.status_code == 422


def test_create_vehicle_empty_manufacturer_returns_422():
    # Empty string is invalid as length must be between 1 and 50
    payload = {
        "manufacturer": "",
        "model": "Civic",
        "year": 2021,
    }
    response = client.post("/vehicles", json=payload)

    # Assert HTTP status code is 422 Unprocessable Entity
    assert response.status_code == 422


def test_create_vehicle_whitespace_only_manufacturer_returns_422():
    # Whitespace-only string should be stripped and rejected as empty
    payload = {
        "manufacturer": "   ",
        "model": "Civic",
        "year": 2021,
    }
    response = client.post("/vehicles", json=payload)

    # Assert HTTP status code is 422 Unprocessable Entity
    assert response.status_code == 422
