import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from app.config import settings
from app.exceptions import DataTransformationError
from typing import Tuple, Dict
import os
import pickle
from app.logger import setup_logger
from scipy.sparse import issparse
logger = setup_logger(__name__)

class DataTransformation:
    def __init__(self):
        self.preprocessor_file_path = os.path.join(settings.PREPROCESSOR_MODEL_DIR, settings.PREPROCESSOR_FILENAME)
        os.makedirs(settings.PREPROCESSOR_MODEL_DIR, exist_ok=True)

    def get_data_transformer_object(self):
        try:
            numerical_features = ['use_of_ip', 'count_dot', 'count_www', 'count_atrate', 'count_dir', 
                                  'count_embed_domain', 'short_url', 'count_percentage', 'count_ques', 
                                  'count_hyphen', 'count_equal', 'url_length', 'count_https', 'count_http', 
                                  'hostname_length', 'fd_length', 'tld_length', 'count_digits', 'count_letters']
            categorical_features = ['abnormal_url', 'sus_url']

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])

            return preprocessor, numerical_features, categorical_features

        except Exception as e:
            logger.error(f"Error in getting data transformer object: {e}")
            raise DataTransformationError(f"Error in getting data transformer object: {str(e)}")

    def initiate_data_transformation(self, train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data completed")
            logger.info(f"Train Dataframe Shape: {train_df.shape}")
            logger.info(f"Test Dataframe Shape: {test_df.shape}")

            preprocessor, numerical_features, categorical_features = self.get_data_transformer_object()

            target_column_name = "type"
            feature_columns = train_df.columns.drop(target_column_name).tolist()

            input_feature_train_df = train_df[feature_columns]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[feature_columns]
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logger.info(f"Preprocessed train array shape: {input_feature_train_arr.shape}")
            logger.info(f"Preprocessed test array shape: {input_feature_test_arr.shape}")

            # Convert to dense array if sparse
            if issparse(input_feature_train_arr):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if issparse(input_feature_test_arr):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Encode target variable
            le = LabelEncoder()
            target_feature_train_arr = le.fit_transform(target_feature_train_df)
            target_feature_test_arr = le.transform(target_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logger.info(f"Final train array shape: {train_arr.shape}")
            logger.info(f"Final test array shape: {test_arr.shape}")

            # Save all components in a single pickle file
            preprocessor_dict = {
                "preprocessor": preprocessor,
                "label_encoder": le,
                "numerical_features": numerical_features,
                "categorical_features": categorical_features
            }

            with open(self.preprocessor_file_path, 'wb') as f:
                pickle.dump(preprocessor_dict, f)

            logger.info(f"Saved preprocessor file at: {self.preprocessor_file_path}")
            logger.info(f"Preprocessor file size: {os.path.getsize(self.preprocessor_file_path) / 1024:.2f} KB")

            # Verify saved data
            with open(self.preprocessor_file_path, 'rb') as f:
                loaded_preprocessor_dict = pickle.load(f)

            logger.info("Preprocessor components:")
            for key, value in loaded_preprocessor_dict.items():
                logger.info(f"  {key}: {type(value)}")

            transformed_train_path = os.path.join(settings.TRAIN_DATA_DIR, "transformed_train_data.npy")
            transformed_test_path = os.path.join(settings.TEST_DATA_DIR, "transformed_test_data.npy")

            np.save(transformed_train_path, train_arr)
            np.save(transformed_test_path, test_arr)

            logger.info(f"Saved transformed train data to {transformed_train_path}")
            logger.info(f"Saved transformed test data to {transformed_test_path}")

            return (
                train_arr,
                test_arr,
                self.preprocessor_file_path,
            )

        except Exception as e:
            logger.error(f"Exception occurred in the initiate_data_transformation: {e}")
            raise DataTransformationError(f"Error occurred in data transformation process: {str(e)}")

    @staticmethod
    def load_preprocessor(preprocessor_file_path: str) -> Dict:
        try:
            with open(preprocessor_file_path, 'rb') as f:
                preprocessor_dict = pickle.load(f)
            logger.info("Loaded preprocessor components successfully.")
            return preprocessor_dict
        except Exception as e:
            logger.error(f"Error loading preprocessor components: {e}")
            raise DataTransformationError(f"Error loading preprocessor components: {str(e)}")