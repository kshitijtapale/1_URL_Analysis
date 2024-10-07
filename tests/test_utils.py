import pytest
import os
from app.utils import save_object, load_object
from app.exceptions import ModelNotFoundError

def test_save_and_load_object():
    test_obj = {"key": "value"}
    file_path = "artifacts\models\model.pkl"
    
    try:
        save_object(file_path, test_obj)
        assert os.path.exists(file_path)
        
        loaded_obj = load_object(file_path)
        assert loaded_obj == test_obj
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_load_nonexistent_object():
    with pytest.raises(ModelNotFoundError):
        load_object("nonexistent_file.pkl")