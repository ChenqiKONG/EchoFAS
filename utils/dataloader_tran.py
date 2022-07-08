import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from PIL import Image
import random

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
        fft = np.load(fft_path)
        label = self.data_path_details.iloc[idx, 1]

        samples = {'ffts': fft,
                  'labels': label}
        if self.transform:
            samples = self.transform(samples)
        return samples

class my_transforms(object):

    def __call__(self, sample):
        ffts = sample['ffts']
        labels = sample['labels']
        ffts = torch.from_numpy(ffts.copy()).float() 
        labels = torch.tensor(labels).long()
        sample_tran = {'ffts': ffts,
                  'labels': labels}
        return sample_tran