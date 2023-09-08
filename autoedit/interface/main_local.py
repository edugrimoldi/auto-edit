import numpy as np
import pandas as pd

from colorama import Fore, Style
import tensorflow as tf

from autoedit.params import *
from autoedit.ml_logic.encoders import load_wav_stereo
from autoedit.ml_logic.preprocessor import preprocess_train, preprocess_predict
from autoedit.ml_logic.model import compile_model, initialize_model, train_model

def train(audio: bytes) -> Model:
    """
    Take a balanced dataset of audios (1 sec. long) in .wav format.
    This method transforms the audio-clips into STFT spectrograms (images) 
    and train a Convolutional Neural Network model.
    """
    print(Fore.MAGENTA + "\n⭐️ Preprocess and train ⭐️" + Style.RESET_ALL)
    
    pos = tf.data.Dataset.list_files(POS+'/*.wav')
    neg = tf.data.Dataset.list_files(NEG+'/*.wav')
    
    # Create column of 0 (negative) or 1 (positive)
    positives = tf.data.Dataset.zip((pos, 
                                     tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
    negatives = tf.data.Dataset.zip((neg, 
                                     tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
    data = positives.concatenate(negatives)
    
    # Preprocess the data and create 16 batchs of N samples.
    data_map = data.map(preprocess_train)
    data_map = data_map.cache()
    data_map = data_map.shuffle(buffer_size=BUFFER_SIZE)
    data_map = data_map.batch(16)
    data = data_map.prefetch(8)

    # Create train and test datasets of test_size 0.3
    train = data_map.take(int(len(data)*0.7))
    test = data_map.skip(int(len(data)*0.7)).take(int(len(data)*0.3))
    
    samples, labels = train.as_numpy_iterator().next()
    shape = samples.shape

    
    # Train a model on the training set, using `model.py`
    model = initialize_model(input_shape=shape[1:])
    
    learning_rate = 0.0005 # Check this hyperparam!
    patience = 8
    
    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(model, 
                                 train_data=train, 
                                 test_data=test,
                                 patience=patience,
                                 validation_data=test)

    # Compute the validation metric
    val_precision = np.max(history.history['val_precision'])

    print("✅ preprocess_and_train() done")
    
    return model
   
   
def pred(data_pred: object) -> pd.DataFrame:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)
    
    if data_pred is None:
       return "I think you should upload a video ;)"
    
    wav = load_wav_stereo(data_pred)
    
    # Slice and preprocess the audio.
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(
        wav, 
        wav, 
        sequence_length=RATE_OUT, 
        sequence_stride=RATE_OUT, 
        batch_size=1)
    audio_slices = audio_slices.map(preprocess_predict)
    audio_slices = audio_slices.batch(64)

    # Change the line below with  "model = load_model()"
    #trained_model = train_model()
    
    yhat = model.pred_model(audio_slices)
    yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
    
    shoot_time = np.array((yhat, list(range(0, len(yhat)))))
    shoot_time= np.transpose(shoot_time)
    
    # Create a dataframe with prediction results
    shoot_time_df = pd.DataFrame(shoot_time,
                              columns=["pred","sec"])
    
    shoot_time_df = shoot_time_df[shoot_time_df["pred"]==1]
    
    print(f"✅ Prediction done")
    return shoot_time_df


if __name__ == '__main__':
    try:
        train()
        pred()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
