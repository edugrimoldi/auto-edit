from colorama import Fore, Style
from typing import Tuple

import tensorflow as tf
from autoedit.ml_logic.encoders import load_wav_stereo, separate_channels, create_stft_spectrogram
from autoedit.params import *


def preprocess_train(file_path: str,
                label: int) -> Tuple[tf.signal.stft, int]:
    print(Fore.BLUE + "\nPreprocessing audio..." + Style.RESET_ALL)

    # Decode the audio file
    wav_total = load_wav_stereo(file_path)

    wav_c1, wav_c2 = separate_channels(wav_total)

    # Fill with a padding of zeros and concatenate
    zero_padding_c1 = tf.zeros(RATE_OUT - len(wav_c1), dtype=tf.float32)
    zero_padding_c2 = tf.zeros(RATE_OUT - len(wav_c2), dtype=tf.float32)
    wav_c1 = tf.concat([zero_padding_c1, wav_c1],0)
    wav_c2 = tf.concat([zero_padding_c2, wav_c2],0)

    # Create a STFT spectrogram for each channel and concatenate
    spectrogram = create_stft_spectrogram(wav_c1, wav_c2)

    print("✅ Spectrogram created succesfully")
    return spectrogram, label


def preprocess_predict(sample, index):
    print(Fore.BLUE + "\nPreprocessing audio..." + Style.RESET_ALL)

    sample= tf.squeeze(sample)

    wav_c1, wav_c2 = separate_channels(sample)
    spectrogram = create_stft_spectrogram(wav_c1, wav_c2)

    return spectrogram
