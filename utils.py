import requests
import ast
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_nan(df, title):
    """
    Parameters:
    - df: Pandas dataframe
    - title: string
    Calculate the percentage of NaN values in each column of the dataframe (df) to get an idea of how much data is missing
    """
    nan_values = df.isnull().mean() * 100
    # Create a bar plot
    nan_values.plot(kind='bar', figsize=(12,6))
    plt.title(title)
    plt.ylabel('Percentage missing values')
    plt.xlabel('Features')
    plt.show()
    
def query_wikidata():
    """
    Args:
    - None
    Returns:
    - JSON data with IMDb ID and Freebase ID from Wikidata
    
    1. Define the URL for the Wikidata SPARQL endpoint
    2. Query items with IMDb ID and Freebase ID
    """
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
    """
    Convert the JSON data to a Pandas Dataframe
    Args:
    - data (JSON data from Wikidata)
    Returns:
    - A Pandas Dataframe containing the mapping between each films' Freebase ID and IMDb ID, this is done in order to join the IMDb dataset and our CMU Movies Summary dataset
     """
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

    Args:
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

def is_missing(val):
    """Check if the value is NaN, an empty list, or zero."""
    return pd.isna(val) or val == [] or val == 0

def plot_missing_values_percentage(df):
    """
    Plots the percentage of missing values in each column of the DataFrame.

    Args:
    - df : Pandas Dataframe

    Returns:
    - None
    """
    # Apply the custom missing value checker to the DataFrame
    missing_values = df.map(is_missing)

    # Calculate the ratio of missing values
    missing_ratio = missing_values.mean().sort_values()

    # Plot the results
    missing_ratio.plot(
        kind="bar", figsize=(15, 4),
        title="Percentage of missing values per feature",
        ylabel="Ratio of missing values per feature"
    )

def plot_data_set(df):
    """
    Plots each column in the DataFrame.

    Args:
    - df : Pandas Dataframe

    Returns:
    - None
    """
    df.plot(lw=0, marker=".", subplots=True, layout=(-1, 4),
          figsize=(15, 30), markersize=1);

def pair_plot_continuous_features(df):
    """
    Creates a pair plot of the continuous numerical features in the DataFrame.
    
    Args:
    - df : Pandas Dataframe

    Returns:
    - None
    """
    # Creates mask to identify numerical features with more or less than 25 unique features
    cols_continuous = df.select_dtypes(include="number").nunique() >= 25
    # Create a new dataframe which only contains the continuous features
    df_continuous = df[cols_continuous[cols_continuous].index]
    df_continuous.shape
    sns.pairplot(df_continuous, height=1.5, plot_kws={"s": 2, "alpha": 0.2});


def plot_top_20_most_popular(df,column_name, xlabel, ylabel, title):
    """
    Plots the 20 most common values.

    Args:
    - df : Pandas Dataframe
    - column_name : string
    - xlabel : string
    - ylabel : string
    - title : string

    Returns:
    - None
    """
    df_exploded = df.explode(column_name)
    values = df_exploded[column_name].value_counts()
    top20_values = values.sort_values(ascending = False)[:20]
    plt.figure(figsize=(20,6))
    top20_values.plot.bar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    


def plot_histograms(df, column_names):
    """
    plots a histogram for each specified column in the DataFrame.

    Args:
    - df : Pandas Dataframe
    - column_names : list of columns to plot

    Returns:
    - None
    """
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


def transform_row(row):
    """
    Transforms a JSON string into a list of its values.

    Args:
    - row (str): A JSON string representing a dictionary.

    Returns:
    - list: A list of values extracted from the JSON string.
    """
    # Load the JSON string into a Python dictionary
    # and then extract its values into a list.
    res = list(json.loads(row).values())

    # Return the list of values.
    return res


def safe_literal_eval(x):
    """
    Tries to evaluate a string x as a python expression.

    Args:
    - x (str): The string to evaluate.

    Returns:
    - list: The evaluated result as a list. If the evaluation fails, returns an empty list.
    """
    try:
        return ast.literal_eval(x)
    except ValueError:
        return []


def get_names(x):
    """
    Extracts the 'name' value.

    Args:
    - x (list): The list of dictionaries to process.

    Returns:
    - list: A list of 'name' values. If the extraction fails, returns an empty list.
    """
    try:
        result = []
        for d in x:
            result.append(d['name'])
        return result
    except TypeError:
        return []


    



