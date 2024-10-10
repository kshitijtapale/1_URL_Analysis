import os
import numpy as np
import joblib
from app.config import settings
from app.exceptions import PredictionError
from app.logger import setup_logger
from app.ml.feature_extraction import FeatureExtractor

logger = setup_logger(__name__)

class URLPredictor:
    def __init__(self):
        self.model_file_path = os.path.join(settings.READY_MODEL_DIR, settings.MODEL_FILENAME)
        self.feature_extractor = FeatureExtractor()
        self.model, self.label_encoder = self.load_model()

    def load_model(self):
        try:
            model, label_encoder = joblib.load(self.model_file_path)
            return model, label_encoder
        except Exception as e:
            logger.error(f"Exception occurred in loading model: {e}")
            raise PredictionError(f"Error occurred in loading model: {str(e)}")

    def predict(self, url: str) -> str:
        try:
            # Extract features
            features = self.feature_extractor.extract_features(url)
            feature_vector = np.array(list(features.values())).reshape(1, -1)

            # Make prediction
            encoded_prediction = self.model.predict(feature_vector)
            prediction = self.label_encoder.inverse_transform(encoded_prediction)

            return prediction[0]
        except Exception as e:
            logger.error(f"Error in URL prediction: {str(e)}")
            raise PredictionError(f"Error in URL prediction: {str(e)}")