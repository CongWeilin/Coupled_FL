import json
import math
import numpy as np
import os
import sys
import random

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

"""
Create data_loader
"""
class synthetic(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.array(X)).float()
        self.y = torch.from_numpy(np.array(y)).long()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        return X, y
    
def train_val_dataloader(X_split, y_split, idx, batch_size):
    """
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    """
    combined = list(zip(X_split[idx], y_split[idx]))
    np.random.shuffle(combined)
    X_split[idx][:], y_split[idx][:] = zip(*combined)

    num_samples = len(X_split[idx])
    train_len = int(0.9 * num_samples)
    test_len = num_samples - train_len

    trainloader = DataLoader(synthetic(X_split[idx][:train_len], y_split[idx][:train_len]),
                             batch_size=batch_size, shuffle=True)
    validloader = DataLoader(synthetic(X_split[idx][train_len:], y_split[idx][train_len:]),
                             batch_size=test_len, shuffle=False)
    return trainloader, validloader

"""
Generate synthetic data, code from FedProx
"""
def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex

#X, y = generate_synthetic(alpha=0, beta=0, iid=0)     # synthetiv (0,0)
#X, y = generate_synthetic(alpha=0.5, beta=0.5, iid=0) # synthetic (0.5, 0.5)
#X, y = generate_synthetic(alpha=1, beta=1, iid=0)     # synthetic (1,1)
#X, y = generate_synthetic(alpha=0, beta=0, iid=1)     # synthetic_IID

def generate_synthetic(alpha, beta, iid, num_user, dimension=60, num_class=10):
    
    samples_per_user = np.random.lognormal(4, 2, (num_user)).astype(int) + 50
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(num_user)]
    y_split = [[] for _ in range(num_user)]


    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, num_user)
    mean_b = mean_W
    B = np.random.normal(0, beta, num_user)
    mean_x = np.zeros((num_user, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(num_user):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, num_class))
        b_global = np.random.normal(0, 1,  num_class)

    for i in range(num_user):

        W = np.random.normal(mean_W[i], 1, (dimension, num_class))
        b = np.random.normal(mean_b[i], 1,  num_class)

        if iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        print("{}-th users has {} exampls".format(i, len(y_split[i])))

    weight_per_user = samples_per_user/num_samples
    return X_split, y_split, weight_per_user
