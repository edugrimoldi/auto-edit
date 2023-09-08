import numpy as np
import pandas as pd
import time

from colorama import Fore, Style
from typing import Tuple

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
        train_data: tf.tensor,
        test_data: tf.tensor,
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

def pred_model(
    model: Model,
    pred_data: tf.tensor) -> tf.tensor:
    
    y_pred = model.predict(pred_data)
    
    return y_pred
    

