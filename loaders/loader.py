import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from . import youtube_loader, lab_loader

class CombinedDataset:
    
    def __init__(self, youtube_root, lab_root, test_root, context_window, val_split=0.05):
        self.youtube_dataset = youtube_loader.YoutubeDataset(youtube_root, context_window)
        self.lab_dataset = lab_loader.LabDataset(lab_root)
        self.test_dataset = youtube_loader.YoutubeDataset(test_root, context_window)
        self._set_mean_and_std()
        self.train_dataset = ConcatDataset([self.youtube_dataset, self.lab_dataset])
        self.train_dataset, self.val_dataset = split_dataset(self.train_dataset, val_split=val_split)
        self.lab_mask = torch.from_numpy(self.lab_dataset.lab_mask).to(torch.float32)
        self.youtube_mask = torch.from_numpy(self.youtube_dataset.youtube_mask).to(torch.float32)
        self.random_mask = torch.from_numpy(self.lab_dataset.lab_mask * self.youtube_dataset.youtube_mask).to(torch.float32)
    
    def _set_mean_and_std(self):
        self.mean, self.std = self.youtube_dataset.mean.copy(), self.youtube_dataset.std.copy()
        self.mean[-15:-1], self.std[-15:-1] = self.lab_dataset.mean.copy()[-15:-1], self.lab_dataset.std.copy()[-15:-1]
        self.youtube_dataset.mean, self.youtube_dataset.std = self.mean, self.std
        self.lab_dataset.mean, self.lab_dataset.std = self.mean, self.std
        self.test_dataset.mean, self.test_dataset.std = self.mean, self.std
    
def collate_fn(batch):
    batch.sort(key=lambda x: x.shape[0], reverse=True)
    lengths = [x.shape[0] for x in batch]
    padded_seqs = pad_sequence(batch, batch_first=True)
    return padded_seqs, torch.tensor(lengths)

def split_dataset(dataset, val_split=0.1):
    num_val = int(len(dataset) * val_split)
    num_train = len(dataset) - num_val
    data_train, data_val = random_split(dataset, [num_train, num_val])
    # loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    # load_val     = DataLoader(data_val,   batch_size=batch_size, shuffle=False,   collate_fn=collate_fn)
    return data_train, data_val