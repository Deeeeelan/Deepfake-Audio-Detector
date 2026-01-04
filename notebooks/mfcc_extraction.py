import numpy as np
import librosa

TARGET_SR = 16000

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=TARGET_SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25)
    m = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([m, std])

def make_spectrogram(path):

    y, sr = librosa.load(path, sr=TARGET_SR)

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
