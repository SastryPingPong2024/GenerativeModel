import os
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

FORMAT_LENGTHS = {
    "p1": 75,
    "p2": 75,
    "p1_root": 3,
    "p2_root": 3,
    "b": 3,
    "p1_pad_hand": 3,
    "p2_pad_hand": 3,
    "p1_pad": 7,
    "p2_pad": 7,
    "fps": 1
}
FORMAT = ["p1", "p1_root", "p2_root", "b", "p1_pad_hand", "p1_pad", "fps"]
FORMAT_RANGES = {}
i = 0
for key in FORMAT:
    FORMAT_RANGES[key] = [i,i+FORMAT_LENGTHS[key]]
    i = i + FORMAT_LENGTHS[key]
FORMAT_SIZE = i

ROOT_JOINT = 8
PADDLE_MASK_INDICES = []
if "p1_pad" in FORMAT:
    PADDLE_MASK_INDICES += list(range(*FORMAT_RANGES["p1_pad"]))
if "p2_pad" in FORMAT:
    PADDLE_MASK_INDICES += list(range(*FORMAT_RANGES["p2_pad"]))
    
class YoutubeDataset(Dataset):
    
    def __init__(self, root_dir, context_window=16):
        self.root_dir = root_dir    
        self.context_window = context_window
        self._load_data()
        self.youtube_mask = np.ones(self.data[0].shape[-1])
        self.youtube_mask[PADDLE_MASK_INDICES] = 0.0
    
    def _load_data(self):
        paths = []
        for match_dir in os.listdir(self.root_dir):
            match_path = os.path.join(self.root_dir, match_dir)
            if os.path.isdir(match_path):
                for file in os.listdir(match_path):
                    if file.endswith('.npy'):
                        paths.append(os.path.join(match_path, file))
        
        self.data = []
        self.masks = []
        for path, mirrored in itertools.product(paths, [False, True]):
            for segment, segment_mask in load_data(path, self.context_window, mirrored=mirrored, mask_non_hitter=True):
                self.data.append(segment)
                self.masks.append(segment_mask)
                
        # Compute statistics for normalization.
        data_concated = np.concatenate(self.data, 0)
        self.mean, self.std = np.mean(data_concated, 0), np.std(data_concated, 0)
        self.token_dim = self.data[0].shape[-1]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        mask = self.masks[index] * self.youtube_mask
        data = (data - self.mean) / (self.std + 1e-8)
        data = data + np.random.randn(*data.shape) / 100
        data[:, FORMAT_RANGES["fps"][0]] = 30.0 / 120.0  # set FPS
        data_and_mask = np.concatenate((data, mask), -1)
        offset = np.random.randint(0, self.context_window-1)
        # data_and_mask = data_and_mask[offset:]
        return torch.from_numpy(data_and_mask).float()
    
    def return_to_raw(self, data, fps=30):
        data = data * (self.std + 1e-8) + self.mean
        data_dict = {}
        for key in FORMAT_LENGTHS:
            if key in FORMAT:
                data_dict[key] = data[:, FORMAT_RANGES[key][0]:FORMAT_RANGES[key][1]].numpy()
            else:
                data_dict[key] = np.zeros((data.shape[0], FORMAT_LENGTHS[key]))
            
        data_dict["p1"] = data_dict["p1"].reshape(data.shape[0], -1, 3)
        data_dict["p2"] = data_dict["p2"].reshape(data.shape[0], -1, 3)
        data_dict["b"]  = data_dict["b"].reshape(data.shape[0], -1, 3)
        data_dict["p1"] = data_dict["p1"] + data_dict["p1_root"].reshape(data.shape[0], -1, 3)
        data_dict["p2"] = data_dict["p2"] + data_dict["p2_root"].reshape(data.shape[0], -1, 3)
        data_dict["p1_pad"][:, :3] = (data_dict["p1_pad"][:, :3] + data_dict["p1_root"]) / 0.003048
        data_dict["p2_pad"][:, :3] = (data_dict["p2_pad"][:, :3] + data_dict["p2_root"]) / 0.003048
        
        raw_data = np.concatenate((np.zeros((data.shape[0], 2, 3)), data_dict["p1"], data_dict["p2"], data_dict["b"]), axis=1) / 0.003048
        raw_data[0, 0, 0] = fps
        raw_data[0, 0, 1] = raw_data[0, 0, 2] = len(raw_data)
        return raw_data, data_dict["p1_pad"],  data_dict["p2_pad"]
        
def load_data(path, context_window, mirrored=False, mask_non_hitter=False):
    """Loads a reconstructed table tennis rally into a dictionary."""
    raw_data = np.load(path)
    raw_data[:, 2:, :] *= 0.003048
    sequence_length = raw_data.shape[0]
    p1_hand, p2_hand = int(raw_data[1, 0, 2]), int(raw_data[2, 0, 0])
    
    # Prune raw_data of any nan segments.
    raw_data = raw_data.reshape(sequence_length, -1)
    nans = np.where(np.any(np.isnan(raw_data), axis=-1) == False)[0]
    raw_data = raw_data[nans[0]:nans[-1]+1]
    sequence_length = raw_data.shape[0]
    raw_data = raw_data.reshape(sequence_length, -1, 3)
    token_masks = np.ones(raw_data.shape)
    
    # Mirror the data if needed
    if mirrored:
        raw_data[:, 2:91, 0] *= -1
        raw_data[:, 2:91, 1] *= -1

    # Form the final data dictionary.
    root_joint = ROOT_JOINT
    data = {
        "sequence_length": sequence_length,
        "p1_hand": p1_hand,
        "p2_hand": p2_hand,
        "p1_hits": raw_data[:, 1, 0],
        "p2_hits": raw_data[:, 1, 1],
        "bounces": raw_data[:, 1, 2],
        "p1":      raw_data[:, 2:27],   # raw_data[:, 2:46], 
        "p2":      raw_data[:, 46:71],  # raw_data[:, 46:90], 
        "b":       raw_data[:, 90:91],  # ball positions,
        "p1_pad":  np.zeros((sequence_length, 7)),
        "p2_pad":  np.zeros((sequence_length, 7)),
        "fps":     np.ones((sequence_length, 1))
    }
    data_mask = {
        "sequence_length": sequence_length,
        "p1":      token_masks[:, 2:27],   
        "p2":      token_masks[:, 46:71],  
        "b":       token_masks[:, 90:91],
        "p1_pad":  np.zeros((sequence_length, 7)),
        "p2_pad":  np.zeros((sequence_length, 7)),
        "fps":     np.ones((sequence_length, 1))
    } 
    
    # Mirroring
    if mirrored:
        data["p1"], data["p2"] = data["p2"], data["p1"]
        data["p1_hand"], data["p2_hand"] = data["p2_hand"], data["p1_hand"]
        data["p1_hits"], data["p2_hits"] = data["p2_hits"], data["p1_hits"]
        data_mask["p1"], data_mask["p2"] = data_mask["p2"], data_mask["p1"]
    
    # Add root information
    data["p1_root"] = data["p1"][:, root_joint:root_joint+1]
    data["p2_root"] = data["p2"][:, root_joint:root_joint+1]   
    data["p1"] = data["p1"] - data["p1_root"]
    data["p2"] = data["p2"] - data["p2_root"]
    data_mask["p1_root"] = data_mask["p1"][:, root_joint:root_joint+1]
    data_mask["p2_root"] = data_mask["p2"][:, root_joint:root_joint+1]
    
    # Add paddle hand information
    data["p1_pad_hand"] = data["p1"][:, p1_hand]
    data["p2_pad_hand"] = data["p2"][:, p2_hand]
    data_mask["p1_pad_hand"] = np.ones(data["p1_pad_hand"].shape)
    data_mask["p2_pad_hand"] = np.ones(data["p2_pad_hand"].shape)
    
    return form_segments(data, data_mask, context_window=context_window, mask_non_hitter=mask_non_hitter)

def format_data(data_dict):
    seq_len = data_dict["sequence_length"]
    data = np.zeros((seq_len, FORMAT_SIZE))
    for key in FORMAT:
        s, e = FORMAT_RANGES[key]
        data[:, s:e] = data_dict[key].reshape(seq_len, -1)
    return data

def form_segments(data, data_mask, context_window=16, mask_non_hitter=False):
    p1_hits = np.where(data["p1_hits"] == 1)[0]
    p2_hits = np.where(data["p2_hits"] == 1)[0]
    hits = np.concatenate((p1_hits, p2_hits), 0)
    hits.sort()
    
    data = format_data(data)
    mask = format_data(data_mask)
    
    # Compile the remaining data for each hit subsequence
    for i in range(1, len(hits)-1):
        segment_start, segment_end = hits[i], hits[i+1]
        segment_start = max(0, segment_start-context_window)
        segment = data[segment_start:segment_end+1].copy()
        segment_mask = mask[segment_start:segment_end+1].copy()
        if mask_non_hitter:
            # if hits[i] in p1_hits:
            #     segment_mask[:, FORMAT_RANGES["p2"][0]:FORMAT_RANGES["p2"][0]]  = 0.0
            #     segment_mask[:, FORMAT_RANGES["p2_pad_hand"][0]:FORMAT_RANGES["p2_pad_hand"][1]] = 0.0
            #     segment_mask[:, FORMAT_RANGES["p2_pad"][0]:FORMAT_RANGES["p2_pad"][1]] = 0.0
            # else:
            #     segment_mask[:, FORMAT_RANGES["p1"][0]:FORMAT_RANGES["p1"][0]]  = 0.0
            #     segment_mask[:, FORMAT_RANGES["p1_pad_hand"][0]:FORMAT_RANGES["p1_pad_hand"][1]] = 0.0
            #     segment_mask[:, FORMAT_RANGES["p1_pad"][0]:FORMAT_RANGES["p1_pad"][1]] = 0.0
            pass
        if hits[i] in p1_hits:
            yield segment, segment_mask
        
if __name__ == "__main__":
    ds = YoutubeDataset("../recons")