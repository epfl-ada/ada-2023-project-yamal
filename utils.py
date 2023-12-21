import requests
import ast
from datetime import datetime
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
    nan_values.plot(kind='bar', figsize=(12, 6))
    plt.title(title)
    plt.ylabel('Percentage missing values')
    plt.xlabel('Features')
    plt.grid(False)
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
    r = requests.get(url, params={'format': 'json', 'query': query})
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
    Prints th number of different values per numerical feature

    Args:
    - df: Pandas DataFrame

    Returns:
    - None
    """
    # For each numerical feature compute number of unique entries
    unique_values = df.select_dtypes(include="number").nunique().sort_values()

    # Plot information with y-axis in log-scale
    unique_values.plot.bar(figsize=(10, 6), title="Unique values per feature")
    plt.grid(False)
    plt.xticks(rotation=0)


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

    plt.grid(False)


def plot_data_set(df):
    """
    Plots each column in the DataFrame.

    Args:
    - df : Pandas Dataframe

    Returns:
    - None
    """
    df.plot(lw=0, marker=".", subplots=True, layout=(-1, 4), figsize=(15, 30), markersize=1)


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
    sns.pairplot(df_continuous, height=1.5, plot_kws={"s": 2, "alpha": 0.2})


def plot_top_20_most_popular(df, column_name, xlabel, ylabel, title):
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
    top20_values = values.sort_values(ascending=False)[:20]
    plt.figure(figsize=(20, 6))
    top20_values.plot.bar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(False)


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
        axes[i].grid(False)

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


def determine_date_format(date_str):
    try:
        # Try to parse the date in yyyy-mm-dd format
        datetime.strptime(date_str, '%Y-%m-%d')
        return 'yyyy-mm-dd'
    except ValueError:
        try:
            # If the first attempt fails, try to parse it in yyyy format
            datetime.strptime(date_str, '%Y')
            return 'yyyy'
        except ValueError:
            # If both attempts fail, the format is neither 'yyyy-mm-dd' nor 'yyyy'
            return 'unknown'


# Helper method that takes as argument actor_names, i.e. the list of the actors that played in a film and returns True
# if and only if one of them has been nominated to the oscars.
def actor_names_in_oscar_actors(actor_names, oscar_actors):
    """
    Returns True if and only if one of the actors in actors_name has been nominated to the oscars.

    Args:
    - actor_names : The list of the actor names that played in a given movie.
    - oscar_actors : The list of the oscar nominated actors.
    """
    for actor_name in actor_names:
        if actor_name in oscar_actors:
            return True
    return False

    # Helper method that takes as argument actor_names, i.e the list of the actors that played in a film and returns
    # the number of oscar nominated actors that played in that movie.


def number_of_oscar_actors(actor_names, oscae_actors):
    count = 0
    for actor_name in actor_names:
        if actor_name in oscae_actors:
            count += 1
    return count


# Function to adjust for inflation
def adjust_for_inflation(row, base_year_cpi):
    """
    Adjusts the box office revenue and budget of a movie for inflation.

    This function takes a row from a DataFrame (representing a movie) and adjusts its
    financial figures (box office revenue and budget) based on the annual CPI (Consumer Price Index)
    to reflect their value in terms of the base year's CPI.

    Args:
        row (pd.Series): A row from a pandas DataFrame. Each row represents a movie and
                        contains data including the annual CPI for its release year,
                        its box office revenue, and its budget.
        base_year_cpi (float): The CPI of the base year against which the adjustment is made.
                                This is the CPI value of the most recent year in the dataset.

    Returns:
        pd.Series: The modified row with adjusted financial figures.
    """

    inflation_factor = base_year_cpi / row['annual_cpi']

    if pd.notna(row['Movie_box_office_revenue']):
        row['Movie_box_office_revenue'] = row['Movie_box_office_revenue'] * inflation_factor
    if pd.notna(row['budget']):
        row['budget'] = row['budget'] * inflation_factor
    return row
