from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Malicious URL Detector"
    MODEL_DIR: str = "artifacts/models"
    DATA_DIR: str = "datasets"
    MODEL_FILENAME: str = "model.pkl"
    PREPROCESSOR_FILENAME: str = "preprocessor.pkl"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()