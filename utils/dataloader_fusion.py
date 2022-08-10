import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class Intra_fusion_datareader(Dataset):
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
        fft = np.load(fft_path)
    
        label = self.data_path_details.iloc[idx, 1]

        samples = {'ffts': fft,
                  'spects':spect,
                  'labels': label}
        if self.transform:
            samples = self.transform(samples)
        return samples

class my_transforms(object):

    def __call__(self, sample):
        labels = sample['labels']
        ffts = sample['ffts']
        spects = sample['spects']
        ### normalize
        ### to pytorch tensor
        ffts = torch.from_numpy(ffts.copy()).float() 
        labels = torch.tensor(labels).long()
        spects = spects.transpose(2, 0, 1)
        spects = torch.from_numpy(spects.copy()).float() 
        ### output
        sample_tran = {
                 'ffts': ffts,
                 'spects': spects,
                  'labels': labels}
        return sample_tran
