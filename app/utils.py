import os
import pickle
from typing import Any
from app.exceptions import ModelNotFoundError
from app.logger import setup_logger

logger = setup_logger(__name__)

def save_object(file_path: str, obj: Any) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logger.error(f"Error in saving object: {e}")
        raise e

def load_object(file_path: str) -> Any:
    try:
        if not os.path.exists(file_path):
            raise ModelNotFoundError(f"The file {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logger.error(f"Error in loading object: {e}")
        raise e