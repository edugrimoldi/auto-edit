from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autoedit.interface.main_local import pred
from autoedit.ml_logic.model import load_model

import os

app = FastAPI()
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model with error handling
try:
    app.state.model = load_model()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading the model: {str(e)}")

# http://127.0.0.1:8000/predict?video=video_number.mp4
@app.get("/predict")
def predict(file):

    return os.path.join("..","file.csv")

    """
    Make a single course prediction.
    """
    """
    if hasattr(app.state, 'model') and app.state.model is not None:
        markers_df = pred(file)
        #markers_df = markers_df.to_json()
        return markers_df
    else:
        return {'error': 'Model not loaded.'}
"""

@app.get("/")
def root():
    return {'greeting': 'Hello'}
