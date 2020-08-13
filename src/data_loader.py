import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import librosa

import config


import os
os.chdir("..")


class Bhajan():
    
    def __init__(self, **kwargs):
        
        options = {'raaga': None,
                   'language': None,
                   'deity': None,
                   'beat': None,
                   'tempo': None}
        options.update(kwargs)
        
        self.raaga = options['raaga']
        self.language = options['language']
        self.deity = options['deity']
        self.beat = options['beat']
        self.tempo = options['tempo']
        
    def resize_array(self, array, length):
        resize_array = np.zeros((array.shape[0], length))
        
        if array.shape[1] >= length:
            resize_array = array[:,:length]
        else:
            resize_array[:,:array.shape[1]] = array
        return resize_array
        
    def load_audio(self, file, **kwargs):
        options = {'sr': 22050,
                   'n_fft': 4096,
                   'n_chroma': 12,
                   'feature_length': 1024
                  }
        options.update(kwargs)
        
        
        y, sr = librosa.load(file)
        self.chromagram = librosa.feature.chroma_stft(y=y, 
                                                      sr=options['sr'], 
                                                      n_fft=options['n_fft'], 
                                                      n_chroma=options['n_chroma'])
        self.chromagram = self.resize_array(self.chromagram, options['feature_length'])

class BhajanDataset(Dataset):
    
    def __init__(self, file_name, **kwargs):
        options = {'target': 'raaga'}
        options.update(kwargs)
        
        df = pd.read_csv(file_name)
        
        if options['target'] not in df.columns:
            raise ValueError(f"Target col {options['target']} not found")
            
        self.df = df
        
        if options['target'] in list(config.labels_map.keys()):
            self.df = self.df.replace({options['target']: config.labels_map[options['target']]})
        else: 
            raise ValueError(f"Target col {options['target']} not found")
        
        self.labels = self.df[options['target']].to_numpy()
        self.training_data = []
        self.n_samples = df.shape[0]
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        bhajan = Bhajan(raaga=row['raaga'],
                        language=row['language'],
                        deity=row['deity'],
                        beat=row['beat'],
                        tempo=row['tempo'])
        
        bhajan.load_audio(row['file_name'])
        chromagram = torch.from_numpy(bhajan.chromagram)

        label = torch.from_numpy(np.asarray(self.labels[index]))
        
        return chromagram, label
    
    def __len__(self):
        return self.df.shape[0]

def get_sampler(data):
    target = data.labels
    counts = np.unique(target, return_counts=True)[1]
    
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    resampling_weights = weights[target]
    sampler = WeightedRandomSampler(resampling_weights, len(data), replacement=True)
    
    return sampler

def get_dataloaders(file_name="data/processed/bhajans_info_cleaned.csv", batch_size=10, lengths=[0.7,0.2,0.1]):
    data = BhajanDataset(file_name=file_name)
    
    train_size = int(lengths[0] * len(data))
    valid_size = int(lengths[1] * len(data))
    test_size = len(data) - train_size - valid_size
    
    train_data, valid_data, test_data = torch.utils.data.random_split(data, [train_size, valid_size, test_size])
    
    # sampler = get_sampler(data)
    
    train_loader = DataLoader(train_data, 
                              # sampler=sampler, 
                              batch_size=batch_size)
    valid_loader = DataLoader(valid_data, 
                              # sampler=sampler, 
                              batch_size=batch_size)
    test_loader = DataLoader(test_data, 
                             # sampler=sampler, 
                             batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader

