import numpy as np
from scipy.special import softmax

import torch
from torch.utils.data import DataLoader, Dataset

class SyntheticDataset(object):

    def __init__(
            self,
            num_classes=2,
            num_tasks=100,
            seed=931231,
            num_dim=10,
            alpha=0,
            beta=0,
            min_num_samples=500,
            max_num_samples=1000,
            test_ratio=0.2):

        np.random.seed(seed)

        self.num_classes = num_classes
        self.num_dim = num_dim
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.beta = beta
        self.min_num_samples = min_num_samples
        self.max_num_samples = max_num_samples
        self.test_ratio = test_ratio
        self.test_data = {'x':[],'y':[]}

        self.Sigma = np.zeros((self.num_dim, self.num_dim))
        for i in range(self.num_dim):
            self.Sigma[i, i] = (i + 1)**(-1.2)
            
        self.num_samples = self.get_num_samples()
        self.tasks = self.get_all_tasks()
        self.train_data = [self.split_task(t) for t in self.tasks]
        self.test_data['x'] = np.concatenate(self.test_data['x'])
        self.test_data['y'] = np.concatenate(self.test_data['y'])
            
    def get_num_samples(self):
        num_samples = np.random.lognormal(3, 2, (self.num_tasks)).astype(int)
        num_samples = [min(s + self.min_num_samples, self.max_num_samples) for s in num_samples]
        return num_samples
    
    def get_all_tasks(self):
        tasks = [self.get_task(s) for s in self.num_samples]
        return tasks

    def get_task(self, num_samples):
        new_task = self._generate_task(num_samples)
        return new_task

    def _generate_x(self, num_samples):
        B = np.random.normal(loc=0.0, scale=self.beta, size=None)
        loc = np.random.normal(loc=B, scale=1.0, size=self.num_dim)

        samples = np.ones((num_samples, self.num_dim + 1))
        samples[:, 1:] = np.random.multivariate_normal(
            mean=loc, cov=self.Sigma, size=num_samples)

        return samples

    def _generate_y(self, x):
        loc = np.random.normal(loc=0, scale=self.alpha, size=None)
        w = np.random.normal(loc=loc, scale=1, size=(self.num_dim + 1,self.num_classes))
        
        num_samples = x.shape[0]
        prob = softmax(np.matmul(x, w) + np.random.normal(loc=loc, scale=1, size=(num_samples, self.num_classes)), axis=1)
                
        y = np.argmax(prob, axis=1)
        return y, w

    def _generate_task(self, num_samples):
        x = self._generate_x(num_samples)
        y, w = self._generate_y(x)

        # now that we have y, we can remove the bias coeff
        x = x[:, 1:]

        return {'x': x, 'y': y, 'w': w}
    
    def split_task(self, task):
        num_samples = task['x'].shape[0]
        shuffle_inds = np.random.permutation(num_samples)
        train_inds = shuffle_inds[:int(num_samples * (1-self.test_ratio))]
        test_inds = shuffle_inds[int(num_samples * (1-self.test_ratio)):]
        train_data = {}
        train_data['x'] = task['x'][train_inds,:].astype('float32')
        train_data['y'] = task['y'][train_inds].astype('long')
        self.test_data['x'].append(task['x'][test_inds,:].astype('float32'))
        self.test_data['y'].append(task['y'][test_inds].astype('long'))
        return train_data
    
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
    
def train_val_dataloader(X_split, y_split, batch_size):
    """
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    """
    combined = list(zip(X_split, y_split))
    np.random.shuffle(combined)
    X_split[:], y_split[:] = zip(*combined)

    num_samples = len(X_split)
    train_len = int(0.9 * num_samples)
    test_len = num_samples - train_len

    trainloader = DataLoader(synthetic(X_split[:train_len], y_split[:train_len]),
                             batch_size=batch_size, shuffle=True)
    validloader = DataLoader(synthetic(X_split[train_len:], y_split[train_len:]),
                             batch_size=test_len, shuffle=False)
    return trainloader, validloader
