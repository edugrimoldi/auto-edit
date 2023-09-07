import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = "1k" # ["1k", "200k", "all"]
CHUNK_SIZE = 200
GCP_PROJECT = "wagon-bootcamp-393215" # TO COMPLETE
GCP_PROJECT_WAGON = "wagon-public-datasets"
BQ_DATASET = "taxifare"
BQ_REGION = "EU"
MODEL_TARGET = "local"

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")

COLUMN_NAMES_RAW = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

DTYPES_RAW = {
    "fare_amount": "float32",
    "pickup_datetime": "datetime64[ns, UTC]",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16"
}

DTYPES_PROCESSED = np.float32

