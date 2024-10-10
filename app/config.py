from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    APP_NAME: str = "Malicious URL Detector"
    DATA_DIR: str = "artifacts"
    READY_MODEL_DIR: str = "artifacts/models/ready"    
    PREPROCESSOR_MODEL_DIR: str = "artifacts/models/preprocessor"    
    RAW_DATASET: str = "artifacts/raw_dataset"    
    EXTRACTED_DATA: str = "artifacts/extracted_data"    
    RAW_DATA_DIR: str = "artifacts/data_ingestion/raw_data"    
    TEST_DATA_DIR: str = "artifacts/data_ingestion/test_data"      
    TRAIN_DATA_DIR: str = "artifacts/data_ingestion/train_data"    
    LOG_DIR: str = "logs"
    MODEL_FILENAME: str = "final_ready_model.pkl"
    PREPROCESSOR_FILENAME: str = "model_preprocessor.pkl"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure log directory exists
os.makedirs(settings.LOG_DIR, exist_ok=True)