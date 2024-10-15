import os
import pickle
from app.exceptions import ModelTrainerError
from app.logger import setup_logger

logger = setup_logger(__name__)

def save_object(file_path, obj):
    """Save an object to a pickle file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error in saving object: {e}")
        raise ModelTrainerError(f"Error occurred while saving object: {str(e)}")

def load_object(file_path):
    """Load an object from a pickle file."""
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logger.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error in loading object: {e}")
        raise ModelTrainerError(f"Error occurred while loading object: {str(e)}")