import os
import pytest
from app.ml.data_ingestion import DataIngestion
from app.exceptions import DataIngestionError

@pytest.fixture
def data_ingestion():
    return DataIngestion()

def test_initiate_data_ingestion(data_ingestion):
    try:
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        assert os.path.exists(train_data_path)
        assert os.path.exists(test_data_path)
    except DataIngestionError as e:
        pytest.fail(f"Data ingestion failed: {str(e)}")

def test_data_ingestion_file_not_found(data_ingestion, monkeypatch):
    def mock_read_csv(*args, **kwargs):
        raise FileNotFoundError("File not found")
    
    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    
    with pytest.raises(DataIngestionError):
        data_ingestion.initiate_data_ingestion()