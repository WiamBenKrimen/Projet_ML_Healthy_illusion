import io

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "sugars_100g": 25.0,
    "fat_100g": 12.0,
    "saturated_fat_100g": 5.0,
    "salt_100g": 0.8,
    "fiber_100g": 1.5,
    "proteins_100g": 4.0,
    "energy_kcal_100g": 450.0,
    "additives_count": 3,
    "main_category": "snacks",
    "country": "france",
    "has_labels": 0,
    "image_saine": 1,
}


def test_required_endpoints():
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    info = client.get("/model/info")
    assert info.status_code == 200
    assert info.json()["trained_at"]


def test_predict_and_validation():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1

    invalid = {**VALID_PAYLOAD, "saturated_fat_100g": 50, "fat_100g": 1}
    assert client.post("/predict", json=invalid).status_code == 422


def test_batch_prediction():
    header = ",".join(VALID_PAYLOAD)
    row = ",".join(str(VALID_PAYLOAD[column]) for column in VALID_PAYLOAD)
    response = client.post(
        "/predict/batch",
        files={"file": ("products.csv", io.BytesIO(f"{header}\n{row}\n".encode()), "text/csv")},
    )
    assert response.status_code == 200
    assert "prediction" in response.text
