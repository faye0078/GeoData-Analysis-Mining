import pandas as pd
from collections import OrderedDict
import numpy as np

def read_data(filename, used_countries):
    csv_covid_data = pd.read_csv(filename)
    country_covid_data = OrderedDict()

    for country in used_countries:
        row_country = csv_covid_data['countryterritoryCode']
        choice = row_country == country
        country_covid_data[country] = csv_covid_data[choice]
    
    return country_covid_data

def preprocessing(data):

    country_data = data
    country_data = country_data[::-1]
    country_data = country_data[~np.isnan(country_data)]

    pre_min_idx = country_data[:country_data.argmax() + 1].argmin()
    clip_country_data = country_data[pre_min_idx : country_data.argmax()+1]

    xp = np.linspace(1, len(clip_country_data), len(clip_country_data))
    xvals = np.linspace(1, len(clip_country_data), 60)
    normalized_data = np.interp(xvals, xp, clip_country_data)

    return normalized_data

def create_dataloader(dataset, look_back=20):
    
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    scalar = max_value - min_value
    dataset = list(map(lambda x: x / scalar, dataset))

    dataX, dataY = [], []
    for i in range(len(dataset)):
        for j in range(len(dataset[i]) - look_back):
            a = dataset[i][j:(j + look_back)]
            dataX.append(a)
            dataY.append(dataset[i][j + look_back])
    data_X = np.array(dataX)
    data_Y = np.array(dataY)

    train_size = int(len(data_X) * 0.8)
    train_X = data_X[:train_size].reshape(-1, 1, look_back)
    train_Y = data_Y[:train_size].reshape(-1, 1, 1)
    test_X = data_X[train_size:].reshape(-1, 1, look_back)
    test_Y = data_Y[train_size:].reshape(-1, 1, 1)
    return {'input': train_X, 'predict': train_Y},  {'input':test_X, 'predict': test_Y}