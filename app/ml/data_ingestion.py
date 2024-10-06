import os
import pandas as pd
from sklearn.model_selection import train_test_split
from app.config import settings
from app.exceptions import DataIngestionError
from typing import Tuple
from app.logger import setup_logger

logger = setup_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join(settings.DATA_DIR, 'raw_data', 'raw_data.csv')
        self.train_data_path = os.path.join(settings.DATA_DIR, 'processed_data', 'train_data.csv')
        self.test_data_path = os.path.join(settings.DATA_DIR, 'processed_data', 'test_data.csv')

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        logger.info("Initiating data ingestion process.")
        try:
            df = pd.read_csv(os.path.join(settings.DATA_DIR, 'preprocessed_data.csv'))
            logger.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path, index=False, header=True)

            logger.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            train_set.to_csv(self.train_data_path, index=False, header=True)

            os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)

            logger.info("Ingestion of the data is completed")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logger.error(f"Exception occurred during Data Ingestion: {e}")
            raise DataIngestionError("Error occurred during data ingestion process")