import numpy as np
import pandas as pd

from google.cloud import bigquery
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from autoedit.params import *
from autoedit.ml_logic.data import clean_data
from autoedit.ml_logic.preprocessor import preprocess_features
from autoedit.ml_logic.registry import save_model, save_results, load_model
from autoedit.ml_logic.model import compile_model, initialize_model, train_model

def preprocess_and_train(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Clean and preprocess data
    - Train a Keras model on it
    - Save the model
    - Compute & save a validation performance metric
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess_and_train" + Style.RESET_ALL)

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """

    # Retrieve `query` data from BigQuery or from `data_query_cache_path` if the file already exists!
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_query_cached_exists = data_query_cache_path.is_file()

    if data_query_cached_exists:
        print("Loading data from local CSV...")
        data = pd.read_csv(data_query_cache_path)

    else:
        print("Loading data from Querying Big Query server...")
        client = bigquery.Client(project=GCP_PROJECT)
        query_job = client.query(query)
        result = query_job.result() 
        data = result.to_dataframe()

        # Save it locally to accelerate the next queries!
        data.to_csv(data_query_cache_path, header=True, index=False)

    # Clean data using data.py
    cleaned_data = clean_data(data)

    # Create (X_train, y_train, X_val, y_val) without data leaks
    # No need for test sets, we'll report val metrics only
    split_ratio = 0.02 # About one month of validation data
    
    train_length = int(len(cleaned_data) * (1- split_ratio))

    data_train = cleaned_data.iloc[:train_length, :].sample(frac=1) # Shuffle datasets to improve training
    data_val = cleaned_data.iloc[train_length:, :].sample(frac=1)

    X_train = data_train.drop("fare_amount", axis=1)
    y_train = data_train[["fare_amount"]]

    X_val = data_val.drop("fare_amount", axis=1)
    y_val = data_val[["fare_amount"]]

    # Create (X_train_processed, X_val_processed) using `preprocessor.py`
    # Luckily, our preprocessor is stateless: we can `fit_transform` both X_train and X_val without data leakage!
    X_train_processed= preprocess_features(X_train)
    X_val_processed = preprocess_features(X_val)

    # Train a model on the training set, using `model.py`
    model = initialize_model(input_shape=X_train_processed.shape[1])
    learning_rate = 0.0005
    batch_size = 256
    patience = 2
    
    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(model, 
                                 X_train_processed, 
                                 y_train, 
                                 batch_size=batch_size, 
                                 patience=patience,
                                 validation_data=(X_val_processed, y_val))

    # Compute the validation metric (min val_mae) of the holdout set
    val_mae = np.min(history.history['val_mae'])

    # Save trained model
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience
    )

    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ preprocess_and_train() done")


def preprocess(min_date: str = '2009-01-01', max_date: str = '2015-01-01') -> None:
    """
    Query and preprocess the raw dataset iteratively (by chunks).
    Then store the newly processed (and raw) data on local hard-drive for later re-use.

    - If raw data already exists on local disk:
        - use `pd.read_csv(..., chunksize=CHUNK_SIZE)`

    - If raw data does not yet exists:
        - use `bigquery.Client().query().result().to_dataframe_iterable()`

    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess by batch" + Style.RESET_ALL)

    from taxifare.ml_logic.data import clean_data
    from taxifare.ml_logic.preprocessor import preprocess_features

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """
    # Retrieve `query` data as dataframe iterable
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")

    data_query_cache_exists = data_query_cache_path.is_file()
    
    if data_query_cache_exists:
        print("Get a dataframe iterable from local CSV...")
        chunks = pd.read_csv(
            data_query_cache_path,
            chunksize=CHUNK_SIZE,
            parse_dates=["pickup_datetime"]
        )

    else:
        print("Get a dataframe iterable from Querying Big Query server...")
        client = bigquery.Client(project=GCP_PROJECT)
        query_job = client.query(query)
        result = query_job.result(page_size=CHUNK_SIZE)
        
        chunks = result.to_dataframe_iterable()

    for chunk_id, chunk in enumerate(chunks):
        print(f"processing chunk {chunk_id}...")

        # Clean chunk
        cleaned_chunk = clean_data(chunk)

        # Create chunk_processed
        # 🎯 Hints: Create (`X_chunk`, `y_chunk`), process only `X_processed_chunk`, then concatenate (X_processed_chunk, y_chunk)
        X_chunk = cleaned_chunk.drop(columns=["fare_amount"])
        y_chunk = cleaned_chunk[["fare_amount"]].to_numpy()
        
        X_processed_chunk = preprocess_features(X_chunk)

        processed_chunk = pd.DataFrame(np.concatenate((X_processed_chunk, y_chunk), axis=1))

        # Save and append the processed chunk to a local CSV at "data_processed_path"
        # 🎯 Hints: df.to_csv(mode=...)
        # 🎯 Hints: We want a CSV without index nor headers (they'd be meaningless)
        processed_chunk.to_csv(data_processed_path,
                                mode='w' if chunk_id==0 else 'a',
                                header=False, 
                                index=False)
            
        # Save and append the raw chunk if not `data_query_cache_exists`
        # 🎯 HINT: only the first chunk should store headers
        if not data_query_cache_exists:
            chunk.to_csv(data_query_cache_path,
                                    mode='w' if chunk_id==0 else 'a',
                                    header=True if chunk_id==0 else False,
                                    index=False)
        
    print(f"✅ data query saved as {data_query_cache_path}")
    print("✅ preprocess() done")
    
    
def train(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    Incremental train on the (already preprocessed) dataset locally stored.
    - Loading data chunk-by-chunk
    - Updating the weight of the model for each chunk
    - Saving validation metrics at each chunks, and final model weights on local disk
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case:train by batch" + Style.RESET_ALL)
    from taxifare.ml_logic.registry import save_model, save_results
    from taxifare.ml_logic.model import (compile_model, initialize_model, train_model)

    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    model = None
    metrics_val_list = []  # store each val_mae of each chunk

    # Iterate in chunks and partial fit on each chunk
    chunks = pd.read_csv(data_processed_path,
                         chunksize=CHUNK_SIZE,
                         header=None,
                         dtype=DTYPES_PROCESSED)

    for chunk_id, chunk in enumerate(chunks):
        print(f"training on preprocessed chunk n°{chunk_id}")
        # You can adjust training params for each chunk if you want!
        learning_rate = 0.0005
        batch_size = 256
        patience=2
        split_ratio = 0.1 # Higher train/val split ratio when chunks are small! Feel free to adjust.

        # Create (X_train_chunk, y_train_chunk, X_val_chunk, y_val_chunk)
        train_length = int(len(chunk)*(1-split_ratio))
        chunk_train = chunk.iloc[:train_length, :].sample(frac=1).to_numpy()
        chunk_val = chunk.iloc[train_length:, :].sample(frac=1).to_numpy()

        X_train_chunk = chunk_train[:, :-1]
        y_train_chunk = chunk_train[:, -1]
        X_val_chunk = chunk_val[:, :-1]
        y_val_chunk = chunk_val[:, -1]

        # Train a model *incrementally*, and store the val MAE of each chunk in `metrics_val_list`
        if model is None:
            model = initialize_model(input_shape=X_train_chunk.shape[1:])
            
        model = compile_model(model, learning_rate)
        
        model, history = train_model(model, 
                                     X_train_chunk,
                                     y_train_chunk,
                                     batch_size=batch_size,
                                     patience=patience,
                                     validation_data=(X_val_chunk, y_val_chunk))
        
        best_val_mae = np.min(history.history['val_mae'])
        metrics_val_list.append(best_val_mae)

    # Return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    # Save model and training params
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        incremental=True,
        chunk_size=CHUNK_SIZE
    )

    print(f"✅ Trained with MAE: {round(val_mae, 2)}")

     # Save results & model
    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ train() done")

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
            pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1],
        ))

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print(f"✅ pred() done")

    return y_pred


if __name__ == '__main__':
    try:
        preprocess_and_train()
        preprocess()
        train()
        pred()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
