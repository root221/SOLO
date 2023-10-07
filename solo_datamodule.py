from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import TensorDataset, DataLoader  
from torchvision import transforms
import pytorch_lightning as pl
import torch
from dataset import BuildDataset, collate_fn
class SoloDataModule(pl.LightningDataModule):
    def __init__(self, data_paths, batch_size=8):
        super().__init__() 
        self.data_paths = data_paths
        self.batch_size = batch_size
        
    def setup(self, stage):
        dataset = BuildDataset(self.data_paths)
        train_len = int(0.8 * len(dataset))
        val_len = int(0.1 * len(dataset))
        test_len = len(dataset) - train_len - val_len
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_len, val_len, test_len]) 
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
