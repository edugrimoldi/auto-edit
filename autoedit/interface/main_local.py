import numpy as np
import pandas as pd

from colorama import Fore, Style
import tensorflow as tf

from autoedit.params import *
from autoedit.ml_logic.encoders import load_wav_stereo
from autoedit.ml_logic.preprocessor import preprocess_train, preprocess_predict
from autoedit.ml_logic.model import *

def process_data() -> tf.Tensor:
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
    data_map = data_map.shuffle(buffer_size=int(BUFFER_SIZE))
    data_map = data_map.batch(16)
    data_map = data_map.prefetch(8)
    
    return data_map

def train(data):

    # Create train and test datasets of test_size 0.3
    train = data.take(int(len(data)*0.7))
    test = data.skip(int(len(data)*0.7)).take(int(len(data)*0.3))
    
    samples, labels = train.as_numpy_iterator().next()
    shape = samples.shape
    
    # Train a model on the training set, using `model.py`
    model = initialize_model(input_shape=shape[1:])
    
    patience = 8
    
    model = compile_model(model)
    model, history = train_model(model, 
                                 train_data=train, 
                                 test_data=test,
                                 patience=patience,
                                 validation_data=test)

    # Compute the validation metric
    #val_precision = np.max(history.history['val_precision'])

    print("✅ Train done")
    
    save_model(model)
    
   
def pred(video: bytes = None) -> pd.DataFrame:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)
    
    if video is None:
        GAME = os.path.join('raw_data', 'Test', 'test5.wav')
        path = os.path.join(os.getcwd(), GAME)
        wav = load_wav_stereo(path)
    
    else:
        wav = load_wav_stereo(video)
    
    print(type(wav))
    # Slice and preprocess the audio.
    sample_slices = tf.keras.utils.timeseries_dataset_from_array(wav, 
                                                                 wav, 
                                                                 sequence_length=8000,         sequence_stride=8000, 
                                                                 batch_size=1)
    
    sample_squeezed = tf.squeeze(sample_slices)
    
    sample_squeezed = sample_slices.map(preprocess_predict)
    sample_squeezed = sample_squeezed.batch(64)

    # Load saved model
    model = load_model()
    
    yhat = pred_model(model, sample_squeezed)
    yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
    
    shoot_time = np.array((yhat, list(range(0, len(yhat)))))
    shoot_time= np.transpose(shoot_time)
    
    # Create a dataframe with prediction results
    shoot_time_df = pd.DataFrame(shoot_time,
                              columns=["pred","sec"])
    
    print(f"✅ Prediction done")
    return shoot_time_df


if __name__ == '__main__':
    try:
        #data_preprocessed = process_data()
        #model = train(data_preprocessed)
        pred()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
