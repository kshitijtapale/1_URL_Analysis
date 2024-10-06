import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from app.config import settings
from app.exceptions import DataTransformationError
from typing import Tuple
import os
import pickle
from app.logger import setup_logger

logger = setup_logger(__name__)

class DataTransformation:
    def __init__(self):
        self.preprocessor_file_path = os.path.join(settings.MODEL_DIR, settings.PREPROCESSOR_FILENAME)

    def get_data_transformer_object(self):
        try:
            categorical_features = ['type']

            categorical_transformer = LabelEncoder()

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", categorical_transformer, categorical_features)
                ],
                remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            logger.error(f"Error in getting data transformer object: {e}")
            raise DataTransformationError("Error in getting data transformer object")

    def initiate_data_transformation(self, train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data completed")

            preprocessor = self.get_data_transformer_object()

            target_column_name = "type_code"
            feature_columns = train_df.columns.drop(target_column_name).tolist()

            input_feature_train_df = train_df[feature_columns]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[feature_columns]
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            os.makedirs(os.path.dirname(self.preprocessor_file_path), exist_ok=True)
            with open(self.preprocessor_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            logger.info("Saved preprocessing object.")

            return train_arr, test_arr, self.preprocessor_file_path

        except Exception as e:
            logger.error(f"Exception occurred in the initiate_data_transformation: {e}")
            raise DataTransformationError("Error occurred in data transformation process")