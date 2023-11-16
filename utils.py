import requests
import numpy as np
import matplotlib.pyplot as plt

# To get an idea of the % of NaNs in each column
def plot_nan(df, title):
    # Calculate the percentage of NaN values in each column
    nan_values = df.isnull().mean() * 100
    # Create a bar plot
    nan_values.plot(kind='bar', figsize=(12,6))
    plt.title(title)
    plt.ylabel('Percentage missing values')
    plt.xlabel('Features')
    plt.show()
    
def query_wikidata():
    url = 'https://query.wikidata.org/sparql'
    query = """
    SELECT ?item ?imdb ?freebase WHERE {
      ?item wdt:P345 ?imdb.
      ?item wdt:P646 ?freebase.
    }
    """
    r = requests.get(url, params = {'format': 'json', 'query': query})
    data = r.json()
    return data

def json_to_df(data):
    imdb_ids = []
    freebase_ids = []
    for item in data['results']['bindings']:
        imdb_ids.append(item['imdb']['value'])
        freebase_ids.append(item['freebase']['value'])
    df = pd.DataFrame({'imdb_id': imdb_ids, 'freebase_id': freebase_ids})
    return df