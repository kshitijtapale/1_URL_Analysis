from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
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
import asyncio

logger = setup_logger(__name__)

app = FastAPI(title="Malicious URL Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: HttpUrl

class PredictionOutput(BaseModel):
    url: str
    prediction: str

def get_url_predictor():
    return URLPredictor()

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
        prediction = await url_predictor.predict(url)  # Assuming URLPredictor.predict is now async
        logger.info(f"Prediction result for {url}: {prediction}")
        return {"url": url, "prediction": prediction}
    except Exception as e:
        logger.error(f"Error in URL prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract_features")
async def extract_features(
    file: Optional[UploadFile] = File(None),
    check_google_index: bool = Form(False),
    bulk_extractor: BulkFeatureExtractor = Depends(get_bulk_extractor)
):
    try:
        output_file = await bulk_extractor.extract_features_from_csv(check_google_index, file)
        # Note: generate_visualizations is commented out, consider making it async if uncommented
        return FileResponse(output_file, filename=os.path.basename(output_file))
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest_data")
async def ingest_data():
    try:
        train_path, test_path = await DataIngestion.create_and_ingest()
        return {
            "message": "Data ingestion completed successfully",
            "train_data_path": train_path,
            "test_data_path": test_path
        }
    except Exception as e:
        logger.error(f"Error in data ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transform_data")
async def transform_data(
    data_transformation: DataTransformation = Depends(get_data_transformation)
):
    try:
        train_path = os.path.join(settings.TRAIN_DATA_DIR, "train_data.csv")
        test_path = os.path.join(settings.TEST_DATA_DIR, "test_data.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise HTTPException(status_code=400, detail="Train or test data not found. Please run data ingestion first.")

        train_arr, test_arr, preprocessor_path = await data_transformation.initiate_data_transformation(train_path, test_path)  # Assuming this method is now async

        # Save transformed data
        transformed_train_path = os.path.join(settings.TRAIN_DATA_DIR, "transformed_train_data.npy")
        transformed_test_path = os.path.join(settings.TEST_DATA_DIR, "transformed_test_data.npy")
        
        await asyncio.gather(
            asyncio.to_thread(np.save, transformed_train_path, train_arr),
            asyncio.to_thread(np.save, transformed_test_path, test_arr)
        )

        return {
            "message": "Data transformation completed successfully",
            "transformed_train_path": transformed_train_path,
            "transformed_test_path": transformed_test_path,
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

        # Load arrays with pickle support (consider making this async)
        train_array, test_array = await asyncio.gather(
            asyncio.to_thread(np.load, transformed_train_path, allow_pickle=True),
            asyncio.to_thread(np.load, transformed_test_path, allow_pickle=True)
        )

        result = await model_trainer.initiate_model_training(train_array, test_array)  # Assuming this method is now async

        return {
            "message": "Model training completed successfully",
            "best_model": result["best_model_name"],
            "model_performance": result["model_report"],
            "model_file_path": result["model_file_path"]
        }
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True) 