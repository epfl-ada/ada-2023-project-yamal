# The Wizardry of Good Movies <img src="https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/44d44b26187031.56352e766c7e0.png" width="50" height="50" alt="Wizards Cap">


## Abstract

The most expensive film ever produced is _Star Wars: The Force Awakens_, costing an estimated 447 million $ , but it ended up being a huge financial success grossing over 2 billion $. What explains this success? One could think that obvious features such as the budget of the film are the main contributors to its success, but this is not always the case. In fact, many other factors influence the success of the film. Our aim is to explore the connection between the films' success (which we measure by its user ratings and revenue when we have enough data) and various less obvious factors such as the gender of the cast, where the film was released etc. Our analysis will aim to find out which factors are the most important and play an important role in contributing to a movies' success.


## Research question <img src="https://static.vecteezy.com/system/resources/thumbnails/000/439/746/small/Basic_Ui__28101_29.jpg" width="30" height="30" alt="Magnifying glass">

Our main research question is straightforward:

- What factors influence a movies' success?

To tackle this wide question, we have chosen a few sub-questions to focus on more specific topics. The aim is to provide a good framework for the project and to give a constructive answer to our research question.


1) Is there a relationship between the movies' release country and its ratings?
2) What is the relationship between a movies' genre and its success?
3) Does the period of release of a movie have an impact on its box office revenue? 
4) What aspects of the movies' cast influence its success?
5) Are longer movies more likely to be more successful?
6) Are movies with higher budgets more successful?
7) what is the relationship between a movies reviews from critics and it’s success?


## Additional datasets

- [TMDB](https://www.kaggle.com/datasets/kakarlaramcharan/tmdb-data-0920) We need additional data on the user ratings and since our base dataset (CMU) does not contain this, we need to add the TMBD dataset which contains the average of all the individual user ratings and the number of votes that the movie received, it also contains information about the revenue of the movie and its budget but many of these values are missing (~90%). The remaining columns in the dataset will most likely not be used.<br>
We used the IMDb IDs in this dataset to merge it with our CMU dataset (this was done by using Wikidata query service to create a mapping from the Freebase IDs to the IMDb IDs) to obtain an augmented version of our base dataset now containing information about the ratings and a bit more information about revenue and the films budgets.


## Methods
### Step 1: General Pre-processing
- We visualise of the data to get insight of the data available to us (number of nan, distributions etc).
- To complete our dataset with the TMDB one, we send a query to [Wikidata](https://query.wikidata.org/) query service to obtain a mapping from the IMDb IDs to the Freebase IDs, this is done in order to correctly merge our base CMU dataset and the TMDB dataset using both IDs as keys.

### Step 2: Feasibility study
- For each sub-question we visualise the data related to it and assess the feasibility of the idea. we analysed sub-questions: 1) | 2) | 3) | 4) as they are the most critical regarding the data that we need to answer them.

### Step 3: Analysing the data (Milestone 3)
- Based on our feasibility study of Milestone 2 we aim to perform:
    - Regressions
    - Hypothesis testing (T test, Chi-squared)
    - Correlation analysis

    and other analysis to answer our questions.

## Proposed timeline

- Week 9: Submit Milestone 2
- Week 10: Each team member tackles a sub-question | Week-end: Meeting to bring together first results to set up the story
- Week 11: Each team member continues working on their sub-question | Week-end: Meeting to discuss about merging results
- Week 12: Merge the different results together to edit the story
- Week 13: Perform final analysis and write the textual description of the work and its results
- Week 14: Finalise the report and discussion, clean the code. Submit.

## Organization within the team

The team organisation will be set up as follows: each team member is assigned one or two sub-questions. To avoid team members having to wait for other people to do their work, each member will work on their own sub-question(s). Recurrent meetings will help us stay on track with our research question and coordinate our activities to make links within sub-questions / individual analysis if needed.