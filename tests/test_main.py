from fastapi.testclient import TestClient
import main

app = main.app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Malicious URL Detector API"}

def test_predict():
    response = client.post("/api/predict", json={"url": "https://example.com"})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_train_model():
    response = client.post("/api/train")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "model_report" in response.json()
    
    
    