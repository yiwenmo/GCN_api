'''
This Python script demonstrates how to create data loaders for a graph dataset using PyTorch Geometric.

It includes functions to split the dataset into training and testing sets and 
to balance the classes within the dataset.

Author: MO, YI WEN
Date: 2024/4/29
'''

import torch
import numpy as np
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from collections import Counter

# 2024/5/24 add
def create_data_loaders(dataset, train_ratio, batch_size):
    # Split dataset into train and test
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class BalancedGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = self.dataset.data
        self.y = self.dataset.data.y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data

    def balance_data(self):
        # Calculate class counts
        class_counts = Counter(self.y.numpy())

        # Find the maximum count
        max_count = max(class_counts.values())

        # Collect indices for each class and resample
        balanced_indices = []
        for label in class_counts.keys():
            indices = np.where(self.y.numpy() == label)[0]
            if len(indices) < max_count:
                indices_to_add = np.random.choice(indices, max_count, replace=True)
            else:
                indices_to_add = np.random.choice(indices, max_count, replace=False)
            balanced_indices.extend(indices_to_add)

        # Shuffle and create a balanced dataset
        np.random.shuffle(balanced_indices)
        
        # Ensure balanced indices are within the range and unique
        balanced_indices = list(set(balanced_indices))
        balanced_indices = [idx for idx in balanced_indices if idx < len(self.dataset)]

        # Use the built-in method to select the subset
        self.dataset = self.dataset.index_select(torch.tensor(balanced_indices))