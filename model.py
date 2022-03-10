import numpy as np
from sklearn.cluster import KMeans
# def PCA():

class myKMeans():
    def __init__(self, classes):
        return 0

def get_KMeans(type, classes):
    if type == 'sklearn':
        return KMeans(n_clusters=classes, random_state = 666)
    elif type == 'mine':
        return myKMeans(classes)

    
