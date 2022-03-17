from tkinter.messagebox import NO
import numpy as np
import random
from sklearn.cluster import KMeans
from torch import nn
# def PCA():

class myKMeans():
    """
    KMeans clustering

    Parameters
    ---
    classes : int
        the number of algo class
    
    Note:
    ---
    The k-means problem is solved using algorithm from Machine Learning written by Zhi-Hua Zhou.
        
    """
    def __init__(self, classes, max_iter=1000):
        self.classes = classes
        self.max_iter = max_iter

    def fit_predict(self, Y):
        sample_idxes = random.sample(range(0, len(Y)), self.classes)
        sample_idxes.sort()
        avg_vec = [None] * len(Y)
        update_vec = [Y[sample_idx] for sample_idx in sample_idxes]
        
        iter = 0
        while iter < self.max_iter:
            avg_vec = update_vec
            predict_indexes = []

            for j in range(len(Y)):
                min_distance = float('inf')
                min_idx = None

                for class_idx in range(len(avg_vec)):
                    distance = np.sqrt(np.sum(np.square(Y[class_idx] - Y[j])))
                    if distance < min_distance:
                        min_distance = distance
                        min_idx = class_idx
                predict_indexes.append(min_idx)

            predict_num = [0] * len(Y)
            for i, predict in enumerate(predict_indexes):
                update_vec[predict] += Y[i]
                predict_num[predict] += 1
            for k in range(len(avg_vec)):
                update_vec[k] /= predict_num[k]
            
            iter += 1

        return predict_indexes

def get_KMeans(type, classes):
    if type == 'sklearn':
        return KMeans(n_clusters=classes, random_state = 666)
    elif type == 'mine':
        return myKMeans(classes, max_iter=200)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers) # rnn
        self.reg = nn.Linear(hidden_size, output_size) # reg

    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

    
