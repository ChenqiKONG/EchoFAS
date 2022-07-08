import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from PIL import Image
import random
import os
from scipy.io import wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

class Intra_echo_datareader(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_path_details = pd.read_csv(csv_file,header=None)
        self.transform = transform  

    def __len__(self):
        return len(self.data_path_details)

    def __getitem__(self, idx):
        np.seterr(divide='ignore',invalid='ignore')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fft_path = self.data_path_details.iloc[idx, 0]
        spect_path = fft_path.split('fft')[0] + 'spect' + fft_path.split('fft')[1]
        spect = np.load(spect_path)
        label = self.data_path_details.iloc[idx, 1]

        samples = {'spects': spect,
                  'labels': label}
        if self.transform:
            samples = self.transform(samples)
        return samples

class my_transforms(object):
    def __call__(self, sample):
        spects = sample['spects']
        labels = sample['labels']
        spects = spects.transpose(2, 0, 1)
        spects = torch.from_numpy(spects.copy()).float() 
        labels = torch.tensor(labels).long()
        ### output
        sample_tran = {'spects': spects,
                  'labels': labels}
        return sample_tran