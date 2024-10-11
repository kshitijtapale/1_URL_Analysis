import os
import pandas as pd
from sklearn.model_selection import train_test_split
from app.config import settings
from app.exceptions import DataIngestionError
from app.logger import setup_logger
import glob

logger = setup_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join(settings.RAW_DATA_DIR, 'raw_data.csv')
        self.train_data_path = os.path.join(settings.TRAIN_DATA_DIR, 'train_data.csv')
        self.test_data_path = os.path.join(settings.TEST_DATA_DIR, 'test_data.csv')

    def initiate_data_ingestion(self):
        logger.info("Initiating data ingestion process.")
        try:
            # Ensure directories exist
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)

            # Find the latest preprocessed CSV file in the EXTRACTED_DATA directory
            extracted_files = glob.glob(os.path.join(settings.EXTRACTED_DATA, "preprocessed_*.csv"))
            if not extracted_files:
                raise FileNotFoundError(f"No preprocessed CSV files found in {settings.EXTRACTED_DATA}")
            
            latest_file = max(extracted_files, key=os.path.getctime)
            logger.info(f"Using preprocessed file: {latest_file}")

            # Read the preprocessed data
            df = pd.read_csv(latest_file)
            logger.info("Read the preprocessed dataset as dataframe")

            # Save raw data (which is now the preprocessed data)
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Saved raw (preprocessed) data at {self.raw_data_path}")

            # Split the dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train data
            train_set.to_csv(self.train_data_path, index=False)
            logger.info(f"Training data saved at {self.train_data_path}")

            # Save test data
            test_set.to_csv(self.test_data_path, index=False)
            logger.info(f"Testing data saved at {self.test_data_path}")

            logger.info("Data ingestion process completed successfully.")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logger.error(f"Exception occurred during Data Ingestion: {e}")
            raise DataIngestionError(f"Error occurred during data ingestion process: {str(e)}")