import os
import pickle
import numpy as np
from typing import Dict
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
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)

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
            features = [self.feature_extractor.extract_features(url) for url in urls]
            features_array = np.array([list(f.values()) for f in features])
            predictions = self.model.predict(features_array)
            return [self.labels[p] for p in predictions]
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise PredictionError(f"Error in batch prediction: {e}")