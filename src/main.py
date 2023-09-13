from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import StreamingResponse

from moviepy.editor import VideoFileClip

from autoedit.interface.main_local import pred
from autoedit.ml_logic.model import load_model

import shutil
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
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Make a single course prediction.
    """
    locals_filename = "video_bonito.mp4"

    with open(locals_filename, 'wb') as f:
        shutil.copyfileobj(file.file, f)
        
    clip = VideoFileClip(locals_filename)
    
    #subclip mean video duration its from the place to start to the end
    audio = clip.audio
    audio.write_audiofile('temp-audio.wav')
    
    markers_df = pred('temp-audio.wav')
    
    return StreamingResponse

    
    """
        #markers_df = markers_df.to_json()
"""

@app.get("/")
def root():
    return {'greeting': 'Hello'}
