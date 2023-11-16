import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def structural_analysis(df):
    """
    Analyze a DataFrame and perform the following tasks:
    1. Print the shape of the DataFrame.
    2. Print the count of each data type in the DataFrame.
    3. Prints th number of different values per numerical feature

    Parameters:
    - df: Pandas DataFrame

    Returns:
    - None
    """
    # Task 1: Print the shape of the DataFrame
    print(f"DataFrame Shape: {df.shape}")

    # Task 2: Print the count of each data type in the DataFrame
    print("\nData Types Counts:")
    print(df.dtypes.value_counts())

    # Task 3: Plot the number of different values for numerical features.

    # For each numerical feature compute number of unique entries
    unique_values = df.select_dtypes(include="number").nunique().sort_values()

    # Plot information with y-axis in log-scale
    unique_values.plot.bar(logy=True, figsize=(15, 4), title="Unique values per feature");

def plot_missing_values_percenatage(df):
    df.isna().mean().sort_values().plot(
    kind="bar", figsize=(15, 4),
    title="Percentage of missing values per feature",
    ylabel="Ratio of missing values per feature");

def plot_data_set(df):
    df.plot(lw=0, marker=".", subplots=True, layout=(-1, 4),
          figsize=(15, 30), markersize=1);

def pair_plot_continuous_features(df):
    # Creates mask to identify numerical features with more or less than 25 unique features
    cols_continuous = df.select_dtypes(include="number").nunique() >= 25
    # Create a new dataframe which only contains the continuous features
    df_continuous = df[cols_continuous[cols_continuous].index]
    df_continuous.shape
    sns.pairplot(df_continuous, height=1.5, plot_kws={"s": 2, "alpha": 0.2});


def plot_top_20_most_popular(df,column_name, xlabel, ylabel, title):
    df_exploded = df.explode(column_name)
    values = df_exploded[column_name].value_counts()
    top20_values = values.sort_values(ascending = False)[:20]
    plt.figure(figsize=(20,6))
    top20_values.plot.bar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    


def plot_histograms(df, column_names):
    num_rows = len(column_names) // 2  # Adjust the number of rows as needed
    num_cols = 2  # Adjust the number of columns as needed

    # Set up subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))

    # Flatten the axes array to handle 1D indexing
    axes = axes.flatten()

    # Plot histograms for each numerical column
    for i, column in enumerate(column_names):
        df[column].hist(ax=axes[i], bins=100)  # Adjust the number of bins as needed
        axes[i].set_title(column)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def split_date(df, dates):
    """
    Define a function to split the date into year and month
    Input: df to modify and its column dates
    """
    movie_release_year = []
    movie_release_month = []
    
    for release_date in dates:
        if '-' in str(release_date):
            splitted_date = release_date.split('-')
            movie_release_year.append(splitted_date[0])
            movie_release_month.append(splitted_date[1])
        else :
            movie_release_year.append(str(release_date)[:4])
            movie_release_month.append(None)
    df['Movie_release_year'] = movie_release_year
    df['Movie_release_month'] = movie_release_month
    df.drop('release_date', axis = 1, inplace = True)

    



