from fastapi import FastAPI, HTTPException
import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wine Classification API",
    description="Predict wine class based on features",
    version="0.1.0",
)

# Define the request body model
class WineRequest(BaseModel):
    # Wine dataset features
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

# Global variable for the model
model = None

@app.on_event("startup")
def load_model():
    """Load the trained model on startup"""
    global model
    
    try:
        # Direct path to the model file
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "mlruns/1/511fc55d79c2457c96decf4ffe14500d/artifacts/better_models/model.pkl"
        )
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load the pickle file directly
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # We'll let the API start but it won't work until a valid model is loaded

@app.get("/")
def home():
    """Root endpoint"""
    return {
        "message": "Wine Classification API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: WineRequest):
    """Predict wine class based on features"""
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame with the format the model expects
        input_data = pd.DataFrame([{
            "alcohol": request.alcohol,
            "malic_acid": request.malic_acid,
            "ash": request.ash,
            "alcalinity_of_ash": request.alcalinity_of_ash,
            "magnesium": request.magnesium,
            "total_phenols": request.total_phenols,
            "flavanoids": request.flavanoids,
            "nonflavanoid_phenols": request.nonflavanoid_phenols,
            "proanthocyanins": request.proanthocyanins,
            "color_intensity": request.color_intensity,
            "hue": request.hue,
            "od280/od315_of_diluted_wines": request.od280_od315_of_diluted_wines,
            "proline": request.proline
        }])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Wine dataset has 3 classes (0, 1, 2)
        wine_classes = ["Class 0", "Class 1", "Class 2"]
        
        # Return prediction result
        result = {
            "prediction": int(prediction[0]),
            "prediction_label": wine_classes[prediction[0]]
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)