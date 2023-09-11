import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autoedit.interface.main_local import pred
from autoedit.ml_logic.model import load_model 

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
        video: bytes
    ):      
    """
    Make a single course prediction.
    """
    markers_df = pred(video)
    
    markers_df = markers_df.to_json()
      
    return {'markers': markers_df}


@app.get("/")
def root():
    return {'greeting': 'Hello'}  
