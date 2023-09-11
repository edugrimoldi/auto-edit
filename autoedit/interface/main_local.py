import numpy as np
import pandas as pd
import datetime

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
        GAME = os.path.join('raw_data', 'Test', 'demotest.wav')
        path = os.path.join(os.getcwd(), GAME)
        wav = load_wav_stereo(path)

    else:
        wav = load_wav_stereo(video)

    # Slice and preprocess the audio.
    sample_slices = tf.keras.utils.timeseries_dataset_from_array(wav,
                                                                 wav,
                                                                 sequence_length=RATE_OUT,
                                                                 sequence_stride=RATE_OUT,
                                                                 batch_size=1)

    sample_slices = sample_slices.map(preprocess_predict)
    sample_slices = sample_slices.batch(64)

    # Load saved model
    model = load_model()

    yhat = pred_model(model, sample_slices)
    yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

    print(f"✅ Prediction done")

    shoot_time = np.array((yhat, list(range(0, len(yhat)))))
    shoot_time= np.transpose(shoot_time)

    # Create a dataframe with prediction results
    shoot_time_df = pd.DataFrame(shoot_time,
                                 columns=["pred","sec"])

    clip_list = {}
    set_start = 0
    set_end = 0
    counter = 0
    shots_wanted = 2
    step_size = 3

    for index, row in shoot_time_df.iterrows():
        if row['pred'] > 0:
            if set_start == 0:
                set_start = index
                counter = 0
            else:
                set_end = index
                counter = 0
        else:
            if counter == 0:
                counter += 1
            elif counter <= step_size and index < len(shoot_time_df) - 1:
                counter += 1
            else:
                counter = 0
                if set_start != 0:
                    shot_count = shoot_time_df['pred'][set_start:set_end + 1].sum()
                    if shot_count >= shots_wanted or index == (len(shoot_time_df)-1):
                        clip_list[set_start] = set_end
                    set_start = 0
                    set_end = 0

    if set_start != 0 and set_end != 0:
        clip_list[set_start] = set_end

    out_data = pd.DataFrame()
    out_data['In'] = list(clip_list.keys())
    out_data['Out'] = list(clip_list.values())

    out_data['In']=out_data['In'].map(lambda x: str(datetime.timedelta(seconds=x)))
    out_data['Out']=out_data['Out'].map(lambda x: str(datetime.timedelta(seconds=x)))
    out_data.set_index('In', inplace=True)
    out_data.to_csv('file.csv', header=None)

    print(f"✅ File saved")


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
