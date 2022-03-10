import pandas as pd
from collections import OrderedDict
import numpy as np

# 读文件，并按国家名称返回Dict类型数据
def read_data(filename, used_countries):
    csv_covid_data = pd.read_csv(filename)
    country_covid_data = OrderedDict()

    for country in used_countries:
        row_country = csv_covid_data['countryterritoryCode']
        choice = row_country == country
        country_covid_data[country] = csv_covid_data[choice]
    
    return country_covid_data

# 将每个国家的数据按一波疫情进行标准化
def normalize(data):
    # 去除nan值
    country_data = data
    country_data = country_data[::-1]
    country_data = country_data[~np.isnan(country_data)]

    # 获取一波疫情数据
    pre_min_idx = country_data[:country_data.argmax() + 1].argmin()
    clip_country_data = country_data[pre_min_idx : country_data.argmax()+1]

    # 插值为长度为60的向量
    xp = np.linspace(1, len(clip_country_data), len(clip_country_data))
    xvals = np.linspace(1, len(clip_country_data), 60)
    normalized_data = np.interp(xvals, xp, clip_country_data)

    return normalized_data