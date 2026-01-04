import numpy as np
import librosa
import soundfile as sf

target_sr = 16000

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=target_sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25)
    m = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([m, std])

def make_spectrogram(path):

    # y, sr = sf.read(path, dtype ='float32') # NOTE: using sf consumes high memory for the FoR dataset
    y, sr = librosa.load(path, sr=target_sr)

    size = sr * 3 # 3 seconds
    y_fixed = librosa.util.fix_length(y, size=size)
    y_norm = librosa.util.normalize(y_fixed) # normalize audio to an amplitude of 1

    mel_spectrogram = librosa.feature.melspectrogram(y=y_norm,
                                                    sr=sr,
                                                    n_fft=2048,
                                                    hop_length=512,
                                                    n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return log_mel_spectrogram
