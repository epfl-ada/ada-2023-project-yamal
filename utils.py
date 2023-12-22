import requests
import ast
from datetime import datetime
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.optimize import linear_sum_assignment



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


 
def sentiment_evalution(plot):
    """
    Function that aims to analyse plot sentiments by examining each sentence of the plot.
    
    Input:  
    - plot: list of the tokens of a movie plot
    
    Output: 
    - sentence_score: list containing the polarity score of each plot sentence 
    - plot_class: string value containing the plot class (either 'Positive', 'Negative', 'Neutral' or nan)

    """

    sentence_class, sentence_score = [], []

    # Loop through all the sentences
    for sentence in plot: 

        sentence_score.append(analyzer.polarity_scores(sentence)['compound'])

        # Positive sentence
        if sentence_score[-1]>= 0.05:
            sentence_class.append(1)

        # Negative sentence
        elif sentence_score[-1] <= -0.05:
            sentence_class.append(-1)
        
        # Neutral sentence
        else:
            sentence_class.append(0)
        
    plot_class = classify_plot(sentence_class)
    
    return pd.Series([sentence_score, plot_class])


def classify_plot(plot_sentence_class):
    """
    Function that aims to classify the plot of a movie into a sentiment class. If 50% of the plot sentences
    belongs to the same sentiment class, this sentiment class is assigned as class of the plot.
    
    Input:  
    - plot_sentence_class: list of sentiment class of the plot sentences
    
    Output:
    - plot_class: string value containing the plot class (either 'Positive', 'Negative' or 'Neutral')

    """

    classes = np.array(plot_sentence_class)
    
    # Positive plot
    if sum(classes==1)/classes.size>=0.5:
        plot_class = 'Positive'

    # Negative plot
    elif sum(classes==-1)/classes.size>0.5:
        plot_class = 'Negative'

    # Neutral plot
    elif sum(classes==0)/classes.size>0.5:
        plot_class = 'Neutral'

    else:
        plot_class = np.nan
        
    return plot_class

def get_similarity(propensity_score1, propensity_score2):
    '''Calculate similarity for instances with given propensity scores'''
    return 1-np.abs(propensity_score1-propensity_score2)

def perform_matching(control_study, treated_study):
    """
    Performs a one to one matching between control group and treatment group based on propensity scores.
    Parameters:
    - control_study: the dataframe representing the control group
    - treated_study: the dataframe representing the treatment group
    Returns :
    - matching : the indices of the matched movies.
    """
    # Initialize the cost matrix with a large value for non-existing edges
    num_control = len(control_study)
    num_treatment = len(treated_study)
    cost_matrix = np.full((num_control, num_treatment), np.inf)

    control_matrix_index = 0
    # Iterate through edges in NetworKit graph and fill the cost matrix
    for control_index, control_row in control_study.iterrows():
        treatment_matrix_index = 0
        for treatment_index, treatment_row in treated_study.iterrows():
            similarity = get_similarity(control_row['Propensity_score'],
                                        treatment_row['Propensity_score'])
            
            # Fill the cost matrix (using negative similarity for maximization)
            cost_matrix[control_matrix_index, treatment_matrix_index] = -similarity
            treatment_matrix_index +=1
        control_matrix_index +=1    

    # Perform the matching using scipy.optimize.linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Translate back to original DataFrame indices
    matching = [(control_study.index[row], treated_study.index[col]) for row, col in zip(row_ind, col_ind)]
    return matching


def categorize_revenue(revenue, threshold):
    return 'blockbuster' if revenue >= threshold else 'non-blockbuster'


def get_unmatched_blockbuster(df, revenue_threshold):
    blockbuster_movies = df.copy().dropna(subset=['Movie_box_office_revenue', 'Movie_runtime'])
    blockbuster_movies['revenue_category'] = blockbuster_movies['Movie_box_office_revenue'].apply(lambda x:categorize_revenue(x,revenue_threshold))

    # Separate into blockbuster and non-blockbuster groups
    blockbuster_group = blockbuster_movies[blockbuster_movies['revenue_category'] == 'blockbuster']
    non_blockbuster_group = blockbuster_movies[blockbuster_movies['revenue_category'] == 'non-blockbuster']

    # Determine the number of movies to sample (minimum count from blockbuster and non-blockbuster groups)
    min_count = min(blockbuster_group.shape[0], non_blockbuster_group.shape[0])

    # Check if there are enough movies for the t-test
    if min_count > 0:
        # Sample the same number of movies from each revenue category
        blockbuster_sample = blockbuster_group.sample(n=min_count, random_state=42)
        non_blockbuster_sample = non_blockbuster_group.sample(n=min_count, random_state=42)

        # Concatenate the balanced data
        unbalanced_data = pd.concat([blockbuster_sample, non_blockbuster_sample])
        
    return unbalanced_data


def get_balanced_blockbuster(df, threshold):
    blockbuster_movies_ = df.copy().dropna(subset=['Movie_box_office_revenue', 'Movie_runtime'])
    blockbuster_movies_['revenue_category'] = blockbuster_movies_['Movie_box_office_revenue'].apply(lambda x: categorize_revenue(x, threshold))
    exploded_blockbuster_ = blockbuster_movies_.explode('genres')

    balanced_data_ = pd.DataFrame(columns=exploded_blockbuster_.columns)

    # Perform t-test for each genre
    for genre_, group_ in exploded_blockbuster_.groupby('genres'):
        # Separate into blockbuster and non-blockbuster groups
        blockbuster_group_ = group_[group_['revenue_category'] == 'blockbuster']
        non_blockbuster_group_ = group_[group_['revenue_category'] == 'non-blockbuster']

        # Determine the number of movies to sample (minimum count from blockbuster and non-blockbuster groups)
        min_count = min(blockbuster_group_.shape[0], non_blockbuster_group_.shape[0])

        # Check if there are enough movies for the t-test
        if min_count > 0:
            # Sample the same number of movies from each revenue category
            blockbuster_sample_ = blockbuster_group_.sample(n=min_count, random_state=42)
            non_blockbuster_sample_ = non_blockbuster_group_.sample(n=min_count, random_state=42)

            # Concatenate the balanced data for this genre
            balanced_data_ = pd.concat([balanced_data_, blockbuster_sample_, non_blockbuster_sample_])
    return balanced_data_

def categorize_budget(budget, threshold):
    return 'low' if budget < threshold else 'high'

def get_matched_buget(df, threshold):
    movies_without_na = df.copy().dropna(subset=['Net profit'])
    movies_without_na['budget_category'] = movies_without_na['budget'].apply(lambda x: categorize_budget(x, threshold))
    exploded_movies = movies_without_na.explode('genres')

    balanced_data = pd.DataFrame(columns=exploded_movies.columns)

    # Perform t-test for each genre
    for genre, group in exploded_movies.groupby('genres'):
        # Separate into low and high budget groups
        low_budget_group = group[group['budget_category'] == 'low']
        high_budget_group = group[group['budget_category'] == 'high']

        # Determine the number of movies to sample (minimum count from low and high budget groups)
        min_count = min(low_budget_group.shape[0], high_budget_group.shape[0])

        # Sample the same number of movies from each budget category
        low_budget_sample = low_budget_group.sample(n=min_count, random_state=42)
        high_budget_sample = high_budget_group.sample(n=min_count, random_state=42)

        # Concatenate the balanced data for this genre
        balanced_data = pd.concat([balanced_data, low_budget_sample, high_budget_sample])
    return balanced_data


    

