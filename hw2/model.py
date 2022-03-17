import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from utils import minValue
class KNN():
    def __init__(self, k):
        self.k_num = k

    def fit(self, X, y):
        self.X = X
        self.y = y
    def predcit(self, input):
        min_value = minValue(self.k_num)
        for i, sample in enumerate(self.X):
            dist = np.sqrt(np.square(sample - input))
            min_value.update(dist, i)

        indexes = min_value.get_values()
        

