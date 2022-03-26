import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from collections import OrderedDict
from utils import minValue, acc_cal
class KNN():
    # using distance weight
    def __init__(self, k):
        self.k_num = k

    def fit(self, X, y):
        self.X = X
        self.y = y
    def predict(self, input):
        min_value = minValue(self.k_num)
        input = input.squeeze()
        for i, sample in enumerate(self.X):
            dist = np.sqrt(np.sum((np.square(sample - input))))
            min_value.update(dist, i)

        values = min_value.min_values
        indexes = min_value.min_indexes
        classes = np.array([self.y[index] for index in indexes])

        u_classes = np.unique(classes)
        classes_weight = OrderedDict()

        tmp = []
        for u_class in u_classes:
            classes_weight[u_class] = (1 / values[(classes == u_class).squeeze()]).sum()
        
        return max(classes_weight, key=classes_weight.get)

class ImageClassificationBase(nn.Module):

    def train_step(self, x, y):
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        a = torch.max(preds, dim=1)
        return loss

    def val_step(self, x, y):
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = acc_cal(preds, y)
        return {'val_acc': acc.item(), 'val_loss': loss}

class SportsClassification(ImageClassificationBase):
    def __init__(self, output_size, pretrained=True):
        super().__init__()
        self.model = models.resnet50(pretrained=pretrained)
        # split
        self.model.fc = nn.Linear(self.model.fc.in_features, output_size)

    def forward(self, x):
        out = self.model(x)
        return out

def get_PCA(length):
    return PCA(n_components=length)

def get_KNN(type, k):
    # mine define and sklearn define
    if type == 'mine':
        return KNN(k)
    elif type == 'sklearn':
        return KNeighborsClassifier(k, weights='distance')

def resnet50(output_size):
    return SportsClassification(output_size)




            
        

