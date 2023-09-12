import os

##################  VARIABLES  ##################
RATE_OUT = int(os.environ.get("RATE_OUT"))
BUFFER_SIZE = int(os.environ.get("BUFFER_SIZE"))

MODEL_NAME = os.environ.get("MODEL_NAME")

POS = os.environ.get("POS")
NEG = os.environ.get("NEG")

GAME = os.environ.get("GAME")

SHOTS_WANTED = int(os.environ.get("SHOTS_WANTED"))
STEP_SIZE = int(os.environ.get("STEP_SIZE"))

##################  CONSTANTS  #####################
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "edugrimoldi", "auto-edit")

