from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from pydantic import BaseModel, HttpUrl
import uvicorn
from typing import List, Dict, Optional
import os
from app.ml.url_predictor import URLPredictor
from app.ml.data_ingestion import DataIngestion
from app.ml.data_transformation import DataTransformation
from app.ml.model_trainer import ModelTrainer
from app.ml.bulk_feature_extraction import BulkFeatureExtractor
from app.config import settings
from app.logger import setup_logger
from app.exceptions import DataIngestionError, DataTransformationError, PredictionError, ModelTrainerError

logger = setup_logger(__name__)

app = FastAPI(title="Malicious URL Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and Response Models
class URLInput(BaseModel):
    url: HttpUrl

class PredictionOutput(BaseModel):
    url: str
    prediction: str

class BatchURLInput(BaseModel):
    urls: List[HttpUrl]

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    
class FeatureExtractionInput(BaseModel):
    check_google_index: bool = False

class FeatureExtractionOutput(BaseModel):
    output_file_path: str

class DataIngestionOutput(BaseModel):
    message: str
    train_data_path: str
    test_data_path: str

class DataTransformationOutput(BaseModel):
    message: str
    transformed_train_path: str
    transformed_test_path: str
    preprocessor_path: str

class ModelTrainingOutput(BaseModel):
    message: str
    best_model: str
    best_model_performance: Dict
    all_models_performance: Dict
    best_model_file_path: str
    comparison_plot_path: str
    confusion_matrix_path: str
    classification_report: str

# Dependency Injection
def get_url_predictor():
    try:
        return URLPredictor()
    except Exception as e:
        logger.error(f"Error initializing URLPredictor: {str(e)}")
        raise HTTPException(status_code=500, detail="Error initializing prediction service")

def get_bulk_extractor():
    return BulkFeatureExtractor()

def get_data_ingestion():
    return DataIngestion()

def get_data_transformation():
    return DataTransformation()

def get_model_trainer():
    return ModelTrainer()

@app.get("/")
async def root():
    return {"message": "Welcome to the Malicious URL Detector API"}

@app.post("/api/predict_url", response_model=PredictionOutput)
async def predict_url(
    url_input: URLInput,
    url_predictor: URLPredictor = Depends(get_url_predictor)
):
    try:
        url = str(url_input.url)
        logger.info(f"Received URL for prediction: {url}")
        prediction = url_predictor.predict(url)
        logger.info(f"Prediction result for {url}: {prediction}")
        return PredictionOutput(url=url, prediction=prediction)
    except PredictionError as e:
        logger.error(f"Error in URL prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in URL prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {str(e)}")

@app.post("/api/predict_batch", response_model=BatchPredictionOutput)
async def predict_batch(
    batch_input: BatchURLInput,
    url_predictor: URLPredictor = Depends(get_url_predictor)
):
    try:
        urls = [str(url) for url in batch_input.urls]
        logger.info(f"Received batch of {len(urls)} URLs for prediction")
        predictions = url_predictor.predict_batch(urls)
        results = [PredictionOutput(url=url, prediction=pred) for url, pred in zip(urls, predictions)]
        logger.info("Batch prediction completed successfully")
        return BatchPredictionOutput(predictions=results)
    except PredictionError as e:
        logger.error(f"Error in batch URL prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in batch URL prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during batch prediction: {str(e)}")

@app.post("/api/predict_urls_batch", response_model=BatchPredictionOutput)
async def predict_urls_batch(
    url_inputs: BatchURLInput,
    url_predictor: URLPredictor = Depends(get_url_predictor)
):
    try:
        urls = [str(url) for url in url_inputs.urls]
        logger.info(f"Received batch of {len(urls)} URLs for prediction")
        predictions = url_predictor.predict_batch(urls)
        results = [PredictionOutput(url=url, prediction=pred) for url, pred in zip(urls, predictions)]
        logger.info(f"Batch prediction completed for {len(urls)} URLs")
        return BatchPredictionOutput(predictions=results)
    except PredictionError as e:
        logger.error(f"Error in batch URL prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in batch URL prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during batch prediction")

@app.post("/api/extract_features", response_model=FeatureExtractionOutput)
async def extract_features(
    file: UploadFile = File(None),
    check_google_index: bool = Form(False),
    bulk_extractor: BulkFeatureExtractor = Depends(get_bulk_extractor)
):
    try:
        output_file = await bulk_extractor.extract_features_from_csv(check_google_index, file)
        return FeatureExtractionOutput(output_file_path=output_file)
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/ingest_data")
async def ingest_data(
    data_ingestion: DataIngestion = Depends(get_data_ingestion),
    
):
    try:
        # if custom_file:
        #     # Save the uploaded file temporarily
        #     temp_file_path = f"temp_{custom_file.filename}"
        #     with open(temp_file_path, "wb") as buffer:
        #         content = await custom_file.read()
        #         buffer.write(content)
            
        #     train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(temp_file_path)
            
        #     # Clean up the temporary file
        #     os.remove(temp_file_path)
        # else:
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        return {
            "message": "Data ingestion completed successfully",
            "train_data_path": train_data_path,
            "test_data_path": test_data_path
        }
    except DataIngestionError as e:
        logger.error(f"Error in data ingestion: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in data ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during data ingestion")

@app.post("/api/transform_data")
async def transform_data(
    data_transformation: DataTransformation = Depends(get_data_transformation)
):
    try:
        train_path = os.path.join(settings.TRAIN_DATA_DIR, "train_data.csv")
        test_path = os.path.join(settings.TEST_DATA_DIR, "test_data.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise HTTPException(status_code=400, detail="Train or test data not found. Please run data ingestion first.")

        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

        return {
            "message": "Data transformation completed successfully",
            "transformed_train_path": os.path.join(settings.TRAIN_DATA_DIR, "transformed_train_data.npy"),
            "transformed_test_path": os.path.join(settings.TEST_DATA_DIR, "transformed_test_data.npy"),
            "preprocessor_path": preprocessor_path
        }
    except Exception as e:
        logger.error(f"Error in data transformation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train_model")
async def train_model(
    model_trainer: ModelTrainer = Depends(get_model_trainer)
):
    try:
        transformed_train_path = os.path.join(settings.TRAIN_DATA_DIR, "transformed_train_data.npy")
        transformed_test_path = os.path.join(settings.TEST_DATA_DIR, "transformed_test_data.npy")

        if not os.path.exists(transformed_train_path) or not os.path.exists(transformed_test_path):
            raise HTTPException(status_code=400, detail="Transformed data not found. Please run data transformation first.")

        train_array = np.load(transformed_train_path, allow_pickle=True)
        test_array = np.load(transformed_test_path, allow_pickle=True)

        best_model = model_trainer.initiate_model_training(train_array, test_array)

        return {
            "message": "Model training completed successfully",
            "best_model": best_model['Model Name'],
            "best_model_accuracy": best_model['accuracy'],
            "best_model_precision": best_model['precision'],
            "best_model_recall": best_model['recall'],
            "best_model_f1_score": best_model['f1_score'],
            "best_model_file_path": os.path.join(settings.READY_MODEL_DIR, f"{best_model['Model Name']}.pkl")
        }
    except ModelTrainerError as e:
        logger.error(f"Error in model training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in model training: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during model training")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)