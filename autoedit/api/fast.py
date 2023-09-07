import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autoedit.ml_logic.preprocessor import preprocess_features
from autoedit.ml_logic.registry import load_model 

app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?video=video_number.mp4
@app.get("/predict")
def predict(
        video: object     # take an object
    ):      
    """
    Make a single course prediction.
    """
    data = video
    
    X = pd.DataFrame(data, index=[0])
    
    model = load_model()
    assert model is not None

    X_processed = preprocess_features(X)
    y_pred = app.state.model.predict(X_processed)
    
    new_data = pd.DataFrame(y_pred, 
                            columns=["Sec", "Number_of_shoots"])
    
    return {new_data}  


@app.get("/")
def root():
    return {'greeting': 'Hello'}  
