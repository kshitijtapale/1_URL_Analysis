class ModelNotFoundError(Exception):
    """Raised when the ML model file is not found."""
    pass

class FeatureExtractionError(Exception):
    """Raised when there's an error in feature extraction."""
    pass

class PredictionError(Exception):
    """Raised when there's an error in making a prediction."""
    pass

class DataIngestionError(Exception):
    """Raised when there's an error in data ingestion process."""
    pass

class DataTransformationError(Exception):
    """Raised when there's an error in data transformation process."""
    pass

class ModelTrainerError(Exception):
    """Raised when there's an error in model training process."""
    pass