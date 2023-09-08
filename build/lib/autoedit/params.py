import os

##################  VARIABLES  ##################
RATE_OUT = os.environ.get("RATE_OUT")
BUFFER_SIZE = os.environ.get("BUFFER_SIZE")

FRAME_LENGTH = os.environ.get("FRAME_LENGTH")
FRAME_STEP = os.environ.get("FRAME_STEP")

MODEL_NAME = os.environ.get("MODEL_NAME")

POS = os.environ.get("POS")
NEG = os.environ.get("NEG")

GAME = os.environ.get("GAME")

##################  CONSTANTS  #####################
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".vae-project", "auto-edit", "training_outputs")

