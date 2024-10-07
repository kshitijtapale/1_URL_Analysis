from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from pydantic import BaseModel, HttpUrl
import uvicorn
from typing import Dict
import os
from app.ml.prediction import URLPredictor
from app.ml.data_ingestion import DataIngestion
from app.ml.data_transformation import DataTransformation
from app.ml.model_trainer import ModelTrainer
from app.ml.bulk_feature_extraction import BulkFeatureExtractor
from app.config import settings
from app.logger import setup_logger

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
    prediction: str

def get_predictor():
    return URLPredictor()

def get_bulk_extractor():
    return BulkFeatureExtractor()

def get_data_ingestion():
    return DataIngestion()

def get_data_transformation():
    return DataTransformation()

@app.get("/")
async def root():
    return {"message": "Welcome to the Malicious URL Detector API"}

@app.post("/api/predict", response_model=PredictionOutput)
async def predict(url_input: URLInput, predictor: URLPredictor = Depends(get_predictor)) -> Dict[str, str]:
    try:
        logger.info(f"Received URL for prediction: {url_input.url}")
        prediction = predictor.predict(str(url_input.url))
        logger.info(f"Prediction result: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def train_model():
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        model_report = model_trainer.initiate_model_trainer(train_arr, test_arr)

        return {"message": "Model training completed successfully", "model_report": model_report}
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract_features")
async def extract_features(
    file: UploadFile = File(...),
    check_google_index: bool = Form(False),
    bulk_extractor: BulkFeatureExtractor = Depends(get_bulk_extractor)
):
    try:
        logger.info(f"Received file for feature extraction: {file.filename}")
        input_file = f"temp_{file.filename}"
        output_file = f"preprocessed_{file.filename}"
        vis_output_dir = "visualizations"

        with open(input_file, "wb") as buffer:
            buffer.write(await file.read())

        await bulk_extractor.extract_features_from_csv(input_file, output_file, check_google_index)
        #bulk_extractor.generate_visualizations(output_file, vis_output_dir)

        os.remove(input_file)  # Clean up temporary file

        return FileResponse(output_file, filename=output_file)
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/ingest_data")
async def ingest_data(
    file: UploadFile = File(...),
    data_ingestion: DataIngestion = Depends(get_data_ingestion)
):
    try:
        logger.info(f"Received file for data ingestion: {file.filename}")
        input_file = f"temp_ingestion_{file.filename}"

        with open(input_file, "wb") as buffer:
            buffer.write(await file.read())

        result = data_ingestion.initiate_data_ingestion(input_file)

        os.remove(input_file)  # Clean up temporary file

        return {
            "message": "Data ingestion completed successfully",
            "file_paths": result
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

        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

        # Save transformed data
        transformed_train_path = os.path.join(settings.TRAIN_DATA_DIR, "transformed_train_data.npy")
        transformed_test_path = os.path.join(settings.TEST_DATA_DIR, "transformed_test_data.npy")
        
        np.save(transformed_train_path, train_arr)
        np.save(transformed_test_path, test_arr)

        return {
            "message": "Data transformation completed successfully",
            "transformed_train_path": transformed_train_path,
            "transformed_test_path": transformed_test_path,
            "preprocessor_path": preprocessor_path
        }
    except Exception as e:
        logger.error(f"Error in data transformation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))