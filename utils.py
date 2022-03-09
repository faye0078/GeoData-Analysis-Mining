import pandas as pd
from collections import OrderedDict

def read_data(filename, used_countries):
    csv_covid_data = pd.read_csv(filename)
    country_covid_data = OrderedDict()

    for country in used_countries:
        row_country = csv_covid_data['countryterritoryCode']
        choice = row_country == country
        country_covid_data[country] = csv_covid_data[choice]
    
    return country_covid_data