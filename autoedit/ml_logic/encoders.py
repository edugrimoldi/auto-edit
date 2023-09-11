from colorama import Fore, Style

import tensorflow as tf
import tensorflow_io as tfio

from autoedit.params import *


def load_wav_stereo(filename):
    print(Fore.BLUE + "\nLoading audio..." + Style.RESET_ALL)
         
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents)
    
    # Removes trailing axis
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    
    # Amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=RATE_OUT)
            
    print("✅ WAV processed",)
    
    return wav


def separate_channels(wav_file):
    print(Fore.GREEN + "\nSeparating in two audio channels..." + Style.RESET_ALL)
    wav_c1 = wav_file[:,0]
    wav_c2 = wav_file[:,1]
    
    # 
    wav_c1 = wav_c1[:RATE_OUT]
    wav_c2 = wav_c2[:RATE_OUT]
    
    return wav_c1, wav_c2


def create_stft_spectrogram(wav_channel_1, wav_channel_2):
    print(Fore.GREEN + "\nCreating a STFT spectrogram..." + Style.RESET_ALL)
    
    spectrogram_c1 = tf.signal.stft(wav_channel_1, frame_length=320, frame_step=32)
    spectrogram_c2 = tf.signal.stft(wav_channel_2, frame_length=320, frame_step=32)
    
    spectrogram_c1 = tf.abs(spectrogram_c1)
    spectrogram_c2 = tf.abs(spectrogram_c2)
    
    spectrogram_c1 = tf.expand_dims(spectrogram_c1, axis=-1)
    spectrogram_c2 = tf.expand_dims(spectrogram_c2, axis=-1)
    spectrogram = tf.concat([spectrogram_c1,spectrogram_c2],axis=-1)
    
    return spectrogram
