import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np


class DYCFCL:
    def __init__(self, batch_size, epochs, train_dataset, groups, dataset_name, client_id=None, device='cuda' ,):
        # Initialize the DY client
        self.criterion_fn = F.cross_entropy
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.groups = groups
        self.current_t = -1
        self.local_epoch = epochs
        self.dataset_name = dataset_name
        self.prototype = {}
        self.client_id = client_id
        self.device =device


    def set_dataloader(self, samples):
        self.train_loader = DataLoader(Subset(self.train_dataset, samples), batch_size=self.batch_size, shuffle=True)


    def set_next_t(self):
        # Sets the dataloader for the next task
        self.current_t += 1
        samples = self.groups[self.current_t]
        self.set_dataloader(samples)