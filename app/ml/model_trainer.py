import os
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from app.config import settings
from app.exceptions import ModelTrainerError
from typing import Dict
import pickle
from app.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(settings.MODEL_DIR, settings.MODEL_FILENAME)

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> Dict[str, float]:
        try:
            logger.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier()
            }

            model_report: Dict[str, float] = {}

            for model_name, model in models.items():
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)

                logger.info(f"{model_name}: Train Score: {train_model_score}, Test Score: {test_model_score}")
                model_report[model_name] = test_model_score

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if model_report[best_model_name] < 0.6:
                raise ModelTrainerError("No best model found")

            logger.info(f"Best found model on both training and testing dataset: {best_model_name}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            with open(self.model_trainer_config.trained_model_file_path, 'wb') as f:
                pickle.dump(best_model, f)

            return model_report

        except Exception as e:
            logger.error(f"Exception occurred in the initiate_model_trainer: {e}")
            raise ModelTrainerError(f"Error occurred in model training process: {e}")