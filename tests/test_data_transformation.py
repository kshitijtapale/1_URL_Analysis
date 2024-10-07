import pytest
import numpy as np
from app.ml.data_transformation import DataTransformation
from app.exceptions import DataTransformationError

@pytest.fixture
def data_transformation():
    return DataTransformation()

def test_get_data_transformer_object(data_transformation):
    try:
        preprocessor = data_transformation.get_data_transformer_object()
        assert preprocessor is not None
    except DataTransformationError as e:
        pytest.fail(f"Failed to get data transformer object: {str(e)}")

def test_initiate_data_transformation(data_transformation):
    # Create mock data for testing
    train_data = np.array([[1, 2, 3, 0], [4, 5, 6, 1], [7, 8, 9, 2]])
    test_data = np.array([[10, 11, 12, 0], [13, 14, 15, 1]])
    
    np.savetxt("mock_train.csv", train_data, delimiter=",")
    np.savetxt("mock_test.csv", test_data, delimiter=",")
    
    try:
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation("mock_train.csv", "mock_test.csv")
        assert isinstance(train_arr, np.ndarray)
        assert isinstance(test_arr, np.ndarray)
    except DataTransformationError as e:
        pytest.fail(f"Data transformation failed: {str(e)}")
    finally:
        import os
        os.remove("mock_train.csv")
        os.remove("mock_test.csv")