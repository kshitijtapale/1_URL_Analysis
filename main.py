from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn
from typing import Dict
from app.ml.prediction import URLPredictor
from app.ml.data_ingestion import DataIngestion
from app.ml.data_transformation import DataTransformation
from app.ml.model_trainer import ModelTrainer
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

predictor = URLPredictor()

class URLInput(BaseModel):
    url: HttpUrl

class PredictionOutput(BaseModel):
    prediction: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Malicious URL Detector API"}

@app.post("/api/predict", response_model=PredictionOutput)
async def predict(url_input: URLInput) -> Dict[str, str]:
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)