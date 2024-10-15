import os
import pandas as pd
from sklearn.model_selection import train_test_split
from app.config import settings
from app.exceptions import DataIngestionError
from app.logger import setup_logger
import glob
from typing import Tuple, Optional

logger = setup_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join(settings.RAW_DATA_DIR, 'raw_data.csv')
        self.train_data_path = os.path.join(settings.TRAIN_DATA_DIR, 'train_data.csv')
        self.test_data_path = os.path.join(settings.TEST_DATA_DIR, 'test_data.csv')

    def _ensure_directories(self):
        """Ensure all necessary directories exist."""
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)

    def _get_latest_preprocessed_file(self) -> str:
        """Find the latest preprocessed CSV file in the EXTRACTED_DATA directory."""
        extracted_files = glob.glob(os.path.join(settings.EXTRACTED_DATA, "preprocessed_*.csv"))
        if not extracted_files:
            raise FileNotFoundError(f"No preprocessed CSV files found in {settings.EXTRACTED_DATA}")
        return max(extracted_files, key=os.path.getctime)

    def initiate_data_ingestion(self, custom_file_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Initiate the data ingestion process.
        
        Args:
            custom_file_path (Optional[str]): Path to a custom CSV file to use instead of the latest preprocessed file.
        
        Returns:
            Tuple[str, str]: Paths to the training and testing data files.
        """
        logger.info("Initiating data ingestion process.")
        try:
            self._ensure_directories()

            if custom_file_path:
                if not os.path.exists(custom_file_path):
                    raise FileNotFoundError(f"Custom file not found: {custom_file_path}")
                input_file = custom_file_path
                logger.info(f"Using custom file: {input_file}")
            else:
                input_file = self._get_latest_preprocessed_file()
                logger.info(f"Using latest preprocessed file: {input_file}")

            df = pd.read_csv(input_file)
            logger.info(f"Read dataset from {input_file}. Shape: {df.shape}")

            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Saved raw data at {self.raw_data_path}")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['type'] if 'type' in df.columns else None)

            train_set.to_csv(self.train_data_path, index=False)
            logger.info(f"Training data saved at {self.train_data_path}. Shape: {train_set.shape}")

            test_set.to_csv(self.test_data_path, index=False)
            logger.info(f"Testing data saved at {self.test_data_path}. Shape: {test_set.shape}")

            logger.info("Data ingestion process completed successfully.")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logger.error(f"Exception occurred during Data Ingestion: {str(e)}")
            raise DataIngestionError(f"Error occurred during data ingestion process: {str(e)}")