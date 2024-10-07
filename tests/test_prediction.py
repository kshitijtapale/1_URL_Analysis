import pytest
from app.ml.prediction import URLPredictor
from app.exceptions import ModelNotFoundError, PredictionError

@pytest.fixture
def url_predictor(monkeypatch):
    def mock_load_model(*args, **kwargs):
        class MockModel:
            def predict(self, X):
                return [0]  # Always predict 'benign' for testing
        return MockModel()
    
    monkeypatch.setattr("app.ml.prediction.URLPredictor._load_model", mock_load_model)
    return URLPredictor()

def test_predict(url_predictor):
    url = "https://www.example.com"
    try:
        prediction = url_predictor.predict(url)
        assert prediction in ['benign', 'defacement', 'phishing', 'malware']
    except PredictionError as e:
        pytest.fail(f"Prediction failed: {str(e)}")

def test_predict_batch(url_predictor):
    urls = ["https://www.example1.com", "https://www.example2.com"]
    try:
        predictions = url_predictor.predict_batch(urls)
        assert len(predictions) == len(urls)
        assert all(pred in ['benign', 'defacement', 'phishing', 'malware'] for pred in predictions)
    except PredictionError as e:
        pytest.fail(f"Batch prediction failed: {str(e)}")

def test_model_not_found(monkeypatch):
    def mock_load_model(*args, **kwargs):
        raise ModelNotFoundError("Model not found")
    
    monkeypatch.setattr("app.ml.prediction.URLPredictor._load_model", mock_load_model)
    
    with pytest.raises(ModelNotFoundError):
        URLPredictor()