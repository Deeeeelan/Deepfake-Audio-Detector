import numpy as np
import librosa

def extract_mfcc(file_path):
    file_path =  file_path
    
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25)
    m = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([m, std])