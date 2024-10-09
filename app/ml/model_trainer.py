import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from app.config import settings
from app.exceptions import ModelTrainerError
from typing import Dict, Any
import joblib
from app.logger import setup_logger
import concurrent.futures
import time
import psutil
import gc

logger = setup_logger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model_file_path = os.path.join(settings.READY_MODEL_DIR, settings.MODEL_FILENAME)
        self.label_encoder = LabelEncoder()

    def make_serializable(self, item):
        if isinstance(item, np.generic):
            return item.item()
        elif isinstance(item, np.ndarray):
            return item.tolist()
        else:
            return item

    def train_and_evaluate_model(self, model_name, model, params, X_train, y_train, X_test, y_test):
        start_time = time.time()
        logger.info(f"Started training {model_name}")
        
        model.set_params(**params)
        model.fit(X_train, y_train)
        
        train_metrics = self.evaluate_model(model, X_train, y_train)
        test_metrics = self.evaluate_model(model, X_test, y_test)
        
        # Perform cross-validation only on a subset of data to save memory
        subset_size = min(10000, len(X_train))
        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]
        cv_scores = cross_val_score(model, X_subset, y_subset, cv=3)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"Finished training {model_name}. Time taken: {training_time:.2f} seconds")
        
        # Clear memory
        del X_subset, y_subset
        gc.collect()
        
        return {
            "model_name": model_name,
            "model": model,
            "train_accuracy": self.make_serializable(train_metrics["accuracy"]),
            "test_accuracy": self.make_serializable(test_metrics["accuracy"]),
            "train_precision": self.make_serializable(train_metrics["precision"]),
            "test_precision": self.make_serializable(test_metrics["precision"]),
            "train_recall": self.make_serializable(train_metrics["recall"]),
            "test_recall": self.make_serializable(test_metrics["recall"]),
            "train_f1": self.make_serializable(train_metrics["f1_score"]),
            "test_f1": self.make_serializable(test_metrics["f1_score"]),
            "train_roc_auc": self.make_serializable(train_metrics["roc_auc"]),
            "test_roc_auc": self.make_serializable(test_metrics["roc_auc"]),
            "cv_mean_accuracy": self.make_serializable(np.mean(cv_scores)),
            "cv_std_accuracy": self.make_serializable(np.std(cv_scores)),
            "training_time": self.make_serializable(training_time)
        }

    def evaluate_model(self, model, X, y):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        try:
            roc_auc = roc_auc_score(y, model.predict_proba(X), multi_class='ovr', average='weighted')
        except AttributeError:
            roc_auc = None
        
        # Clear memory
        del y_pred
        gc.collect()
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

    def check_memory(self, threshold=90):
        memory = psutil.virtual_memory()
        if memory.percent > threshold:
            logger.warning(f"Memory usage is very high: {memory.percent}%. Switching to sequential processing.")
            return False
        return True

    def initiate_model_training(self, train_array: np.ndarray, test_array: np.ndarray) -> Dict[str, Any]:
        try:
            logger.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Encode labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)

            models = {
                "Random Forest": RandomForestClassifier(n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
                "XGBoost": XGBClassifier(n_jobs=-1),
                "LightGBM": LGBMClassifier(n_jobs=-1)
            }

            params = {
                "Random Forest": {'n_estimators': 100, 'max_depth': 10},
                "Gradient Boosting": {'n_estimators': 100, 'learning_rate': 0.1},
                "Logistic Regression": {'C': 1, 'penalty': 'l2'},
                "XGBoost": {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
                "LightGBM": {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31}
            }

            model_results = []
            if self.check_memory(threshold=90):
                with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                    futures = []
                    for model_name, model in models.items():
                        future = executor.submit(
                            self.train_and_evaluate_model,
                            model_name, model, params[model_name],
                            X_train, y_train_encoded, X_test, y_test_encoded
                        )
                        futures.append(future)

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            model_results.append(result)
                        except Exception as e:
                            logger.error(f"An error occurred during parallel processing for a model: {str(e)}")
                            # Continue with other models instead of breaking

                if not model_results:
                    logger.warning("Parallel processing failed for all models. Switching to sequential.")
                    raise Exception("Parallel processing failed. Switching to sequential.")
            else:
                raise Exception("Memory usage is high. Switching to sequential processing.")

        except Exception as e:
            logger.warning(f"Parallel processing not possible: {str(e)}. Using sequential processing.")
            model_results = []
            for model_name, model in models.items():
                try:
                    result = self.train_and_evaluate_model(
                        model_name, model, params[model_name],
                        X_train, y_train_encoded, X_test, y_test_encoded
                    )
                    model_results.append(result)
                except Exception as model_error:
                    logger.error(f"Error training {model_name}: {str(model_error)}")

        if not model_results:
            raise ModelTrainerError("All model training attempts failed.")

        model_report = {result["model_name"]: {k: self.make_serializable(v) for k, v in result.items() if k != "model"} for result in model_results}

        best_model_name = max(model_report, key=lambda x: model_report[x]['test_accuracy'])
        best_model = next(result["model"] for result in model_results if result["model_name"] == best_model_name)

        logger.info(f"Best model found: {best_model_name}")
        logger.info(f"Model report: {model_report[best_model_name]}")

        os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)
        joblib.dump((best_model, self.label_encoder), self.model_file_path)

        # Clear memory
        del X_train, y_train, X_test, y_test, y_train_encoded, y_test_encoded
        gc.collect()

        return {
            "best_model_name": best_model_name,
            "model_report": model_report[best_model_name],
            "model_file_path": self.model_file_path
        }

    @staticmethod
    def load_model(model_path: str):
        try:
            model, label_encoder = joblib.load(model_path)
            return model, label_encoder
        except Exception as e:
            logger.error(f"Exception occurred in loading model: {e}")
            raise ModelTrainerError(f"Error occurred in loading model: {str(e)}")