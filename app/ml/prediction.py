import os
import joblib  # Use joblib instead of pickle
import numpy as np
from typing import List
from app.ml.feature_extraction import FeatureExtractor
from app.config import settings
from app.exceptions import ModelNotFoundError, PredictionError
from app.logger import setup_logger

logger = setup_logger(__name__)

class URLPredictor:
    def __init__(self):
        self.model = self._load_model()
        self.feature_extractor = FeatureExtractor()
        self.labels = ['benign', 'defacement', 'phishing', 'malware']

    def _load_model(self):
        model_path = os.path.join(settings.MODEL_DIR, settings.MODEL_FILENAME)
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise ModelNotFoundError(f"Model file not found at {model_path}")

        # Load the model using joblib
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelNotFoundError(f"Error loading model: {e}")

    def predict(self, url: str) -> str:
        try:
            features = self.feature_extractor.extract_features(url)
            features_array = np.array(list(features.values())).reshape(1, -1)
            prediction = self.model.predict(features_array)
            return self.labels[prediction[0]]
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise PredictionError(f"Error in prediction: {e}")

    def predict_batch(self, urls: List[str]) -> List[str]:
        try:
            # Extract features for each URL
            features = [self.feature_extractor.extract_features(url) for url in urls]
            features_array = np.array([list(f.values()) for f in features])  # Create an array of features
            predictions = self.model.predict(features_array)  # Predict using the model
            return [self.labels[p] for p in predictions]  # Map predictions to labels
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise PredictionError(f"Error in batch prediction: {e}")

if __name__ == "__main__":
    predictor = URLPredictor()
    
    # Test a single URL
    try:
        result = predictor.predict("http://example.com")
        print(f"Prediction for single URL: {result}")
    except PredictionError as e:
        print(f"Prediction error: {e}")

    # Test batch prediction
    try:
        urls = ["http://example.com", "http://malicious-site.com"]
        batch_results = predictor.predict_batch(urls)
        print(f"Batch predictions: {batch_results}")
    except PredictionError as e:
        print(f"Batch prediction error: {e}")

