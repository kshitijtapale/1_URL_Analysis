import pytest
import numpy as np
from app.ml.model_trainer import ModelTrainer
from app.exceptions import ModelTrainerError

@pytest.fixture
def model_trainer():
    return ModelTrainer()

def test_initiate_model_trainer(model_trainer):
    # Create mock data for testing
    train_array = np.array([[1, 2, 3, 0], [4, 5, 6, 1], [7, 8, 9, 2]])
    test_array = np.array([[10, 11, 12, 0], [13, 14, 15, 1]])
    
    try:
        model_report = model_trainer.initiate_model_trainer(train_array, test_array)
        assert isinstance(model_report, dict)
        assert len(model_report) > 0
    except ModelTrainerError as e:
        pytest.fail(f"Model training failed: {str(e)}")

def test_initiate_model_trainer_with_insufficient_data(model_trainer):
    # Create mock data with insufficient samples
    train_array = np.array([[1, 2, 3, 0]])
    test_array = np.array([[4, 5, 6, 1]])
    
    with pytest.raises(ModelTrainerError):
        model_trainer.initiate_model_trainer(train_array, test_array)