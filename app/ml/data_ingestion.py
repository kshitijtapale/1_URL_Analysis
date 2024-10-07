import os
import pandas as pd
from sklearn.model_selection import train_test_split
from app.config import settings
from app.exceptions import DataIngestionError
from typing import Tuple, Dict
from app.logger import setup_logger

logger = setup_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join('artifacts', 'raw_data', 'raw_data.csv')
        self.train_data_path = os.path.join('artifacts', 'train_data', 'train_data.csv')
        self.test_data_path = os.path.join('artifacts', 'test_data', 'test_data.csv')

    def initiate_data_ingestion(self, input_file: str) -> Dict[str, str]:
        logger.info("Initiating data ingestion process.")
        try:
            # Read the input CSV file
            df = pd.read_csv(input_file)
            logger.info(f"Read the dataset from {input_file}")

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Raw data saved at {self.raw_data_path}")

            # Split the dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train data
            train_set.to_csv(self.train_data_path, index=False)
            logger.info(f"Training data saved at {self.train_data_path}")

            # Save test data
            test_set.to_csv(self.test_data_path, index=False)
            logger.info(f"Testing data saved at {self.test_data_path}")

            logger.info("Data ingestion process completed successfully.")

            return {
                "raw_data_path": self.raw_data_path,
                "train_data_path": self.train_data_path,
                "test_data_path": self.test_data_path
            }

        except Exception as e:
            logger.error(f"Exception occurred during Data Ingestion: {e}")
            raise DataIngestionError(f"Error occurred during data ingestion process: {str(e)}")