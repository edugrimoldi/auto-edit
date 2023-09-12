import os
import glob
import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

from autoedit.params import *

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate: float) -> Model:
    """
    Compile the Convolutional Neural Network
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss='BinaryCrossentropy',
        optimizer=optimizer,
        metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()]
    )

    print("✅ Model compiled")

    return model


def train_model(
        model: Model,
        train_data: tf.Tensor,
        test_data: tf.Tensor,
        patience=8,
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
                    train_data,
                    epochs=30,
                    validation_data=test_data,
                    callbacks=[es])

    print(f"✅ Model trained with precision: {round(np.max(history.history['val_precision']), 2)}")

    return model, history


def pred_model(model: Model, pred_data: tf.Tensor) -> tf.Tensor:
    """
    Perform a prediction and return a tf.Tensor
    """
    y_pred = model.predict(pred_data)
    return y_pred


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{MODEL_NAME}.h5"
    """
    #timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH,"models", f"{MODEL_NAME}.h5")
    model.save(model_path)

    print("✅ Model saved locally")


def load_model() -> keras.Model:
    """
    Return a locally saved model (latest one in alphabetical order)
    """
    print(Fore.BLUE + f"\nLoading latest model..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, 'models')
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    print("✅ Model loaded from local disk")

    return latest_model
