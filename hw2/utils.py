import numpy as np
import os
import torch
from PIL import Image
from torchvision.datasets import ImageFolder 
from torchvision import transforms as T
from torch.utils.data import DataLoader
class minValue():
    def __init__(self, min_num):
        self.min_values = np.zeros(min_num)
        self.min_indexes= np.array([None]*min_num)

    def update(self, n_value, n_index):
        for i, value in enumerate(self.min_values):
            if n_value <= value:
                if i == len(self.min_indexes) -1 or n_value >= self.min_values[i + 1]:
                    self.min_values[i] = n_value
                    self.min_indexes[i] = n_index
                    break
            else:
                continue

def get_sport_dataloader(batch_size):
    std=torch.Tensor([0.4687, 0.4667, 0.4540])
    mean=torch.Tensor([0.2792, 0.2717, 0.2852])
    t_trans=T.Compose([
                    T.RandomCrop((196,196),padding=4,padding_mode='reflect'),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=(0, 180)),
                    T.ToTensor(),
                    T.Normalize(mean,std)])
    v_trans=T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean,std)])
    train_set=ImageFolder('../dataset/sports_dataset/train',transform=t_trans)
    val_set=ImageFolder('../dataset/sports_dataset/valid',transform=v_trans)
    test_set=ImageFolder("../dataset/sports_dataset/test",transform=v_trans)
    print(len(train_set),len(val_set), len(test_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader