import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from app.config import settings
from app.exceptions import DataTransformationError
from typing import Tuple
import os
import pickle
from app.logger import setup_logger

logger = setup_logger(__name__)

class DataTransformation:
    def __init__(self):
        self.preprocessor_file_path = os.path.join(settings.PREPROCESSOR_MODEL_DIR, settings.PREPROCESSOR_FILENAME)

    def get_data_transformer_object(self):
        try:
            numerical_features = ['use_of_ip', 'count_dot', 'count_www', 'count_atrate', 'count_dir', 
                                  'count_embed_domain', 'short_url', 'count_percentage', 'count_ques', 
                                  'count_hyphen', 'count_equal', 'url_length', 'count_https', 'count_http', 
                                  'hostname_length', 'fd_length', 'tld_length', 'count_digits', 'count_letters']
            categorical_features = ['abnormal_url', 'sus_url']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            logger.error(f"Error in getting data transformer object: {e}")
            raise DataTransformationError(f"Error in getting data transformer object: {str(e)}")

    def initiate_data_transformation(self, train_path: str, test_path: str) -> Tuple[str, str, str]:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data completed")

            preprocessor = self.get_data_transformer_object()

            target_column_name = "type"
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

            transformed_train_path = os.path.join(settings.TRAIN_DATA_DIR, "transformed_train_data.npy")
            transformed_test_path = os.path.join(settings.TEST_DATA_DIR, "transformed_test_data.npy")

            # Save arrays with pickle protocol
            np.save(transformed_train_path, train_arr, allow_pickle=True)
            np.save(transformed_test_path, test_arr, allow_pickle=True)

            os.makedirs(os.path.dirname(self.preprocessor_file_path), exist_ok=True)
            with open(self.preprocessor_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            logger.info("Saved preprocessing object and transformed data.")

            return (
                transformed_train_path,
                transformed_test_path,
                self.preprocessor_file_path,
            )

        except Exception as e:
            logger.error(f"Exception occurred in the initiate_data_transformation: {e}")
            raise DataTransformationError(f"Error occurred in data transformation process: {str(e)}")