import numpy as np
import librosa

train_files = '../../data/ASVspoof_Dataset/ASVspoof2019_LA_train/flac/'

def extract_mfcc(file_name):
    file_path =  train_files + file_name + ".flac"
    
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25)
    m = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([m, std])