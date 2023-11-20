import itertools

import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn import datasets
from sklearn.model_selection import train_test_split

import nam_bench.datasets.utils.motions as motions


class Digits(Dataset):
    def __init__(self, config):
        sample_size = 10 * 10
        digits = datasets.load_digits()
        _, self.data, _, self.target = train_test_split(
            digits.data,
            digits.target,
            test_size=sample_size,
            random_state=0,
            stratify=digits.target,
        )
        self.data = self.data.astype(np.float32).reshape(-1, 8, 8)
        self.target = self.target.astype(np.float32)
        if hasattr(config, "transforms"):
            self.transform = config.transforms
        else:
            self.transform = None
        self.metainfo = [{"target": str(int(target))} for target in self.target]
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index):
        X = torch.from_numpy(self.data[index])
        X.unsqueeze_(0)
        y = self.target[index]
        
        if self.transform is not None:
            X = self.transform(X)
        
        return X, y


class DigitsForInpainting(Dataset):
    def __init__(self, config):
        sample_size = 10 * 10
        digits = datasets.load_digits()
        _, self.data, _, self.target = train_test_split(
            digits.data,
            digits.target,
            test_size=sample_size,
            random_state=0,
            stratify=digits.target,
        )
        self.mask_size = config.mask_size
        self.data = self.data.astype(np.float32).reshape(-1, 8, 8)
        self.target = self.target.astype(np.float32)
        if hasattr(config, "transforms"):
            self.transform = config.transforms
        else:
            self.transform = None
        self.metainfo = [{"target": str(int(target))} for target in self.target]
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index):
        X = torch.from_numpy(self.data[index])
        y = torch.from_numpy(self.data[index])
        h, w = X.shape
        not_mask_h, not_mask_w = h - self.mask_size, w - self.mask_size
        mask_h = not_mask_h // 2
        mask_w = not_mask_w // 2
        X[mask_h:mask_h+self.mask_size, mask_w:mask_w+self.mask_size] = 0.0

        X.unsqueeze_(0)
        y.unsqueeze_(0)
        
        if self.transform is not None:
            X = self.transform(X)
            y = self.transform(y)
        
        return X, y


class MaskedMovingDigits(Dataset):
    def __init__(self, config):
        np.random.seed(0)
        sample_size = 10 * 10
        digits = datasets.load_digits()
        _, objects, _, target = train_test_split(
            digits.data,
            digits.target,
            test_size=sample_size,
            random_state=0,
            stratify=digits.target,
        )
        
        objects = objects.astype(np.float32).reshape(-1, 8, 8)
        speeds = config.speeds
        thetas = config.thetas
        binarize = config.binarize
        normalize = config.normalize
        
        if normalize:
            objects = objects / 15
            if binarize:
                objects = (objects > 0.5).astype(np.float32)
        elif binarize:
            objects = (objects > 8).astype(np.float32)
        
        batch_of_sequences = []
        ys = [target] * len(speeds) * len(thetas)
        ys = np.concatenate(ys, axis=0)
        self.metainfo = [{"target": str(int(y))} for y in ys]

        for i, (speed, theta) in enumerate(itertools.product(speeds, thetas)):
            batch_of_sequences.append(
                motions.get_sequences(
                    seq_length=config.seq_length,
                    img_size=config.img_size,
                    speed=speed,
                    theta=theta,
                    objects=objects,
                )
            )
            for j in range(len(target)):
                self.metainfo[i * len(target) + j]["speed"] = speed
                self.metainfo[i * len(target) + j]["theta"] = theta
        
        batch_of_sequences = np.concatenate(batch_of_sequences, axis=0)
        
        self.target = batch_of_sequences
        
        masked_batch_of_sequences = np.copy(batch_of_sequences)
        masked_batch_of_sequences[:, -1] = 0.0
        self.data = masked_batch_of_sequences

        if hasattr(config, "transforms"):
            self.transform = config.transforms
        else:
            self.transform = None

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = torch.from_numpy(self.data[index])
        y = torch.from_numpy(self.target[index])
        
        X.unsqueeze_(1)
        y.unsqueeze_(1)

        if self.transform is not None:
            X = self.transform(X)
        
        return X, y