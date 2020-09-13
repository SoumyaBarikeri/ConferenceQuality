"""
This file calculates adjective percentage in Call for Paper text for all conferences
"""
import pandas as pd
import nltk
from collections import Counter
import time


start_time = time.time()

# Get WikiCfp conference data as Dataframe and set index
df = pd.read_csv('../data/wCfP_data_full_new.csv')
df = df.reindex(columns=['eventID', 'text'])
df = df.set_index('eventID')


def adjective_freq(text):
    """
    Function to get percentage of adjectives in conference text

    Parameters
    ----------
    text
        Call for Paper text from WikiCfp

    Returns
    -------
    float
        Percentage of adjectives in text

    """
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    counts = Counter(tag for word, tag in tagged)

    # Calculate percentage of adjectives
    adjective_percent = (counts['JJ'] / len(tagged)) * 100
    adjective_percent = round(adjective_percent, 2)
    return adjective_percent


# Get adjective percentage of each conference by looping over rows of Dataframe and add new feature
for index, row in df.iterrows():
    result = adjective_freq(row['text'])
    print(result)
    df.at[index, 'adj_percent'] = result

df = df.drop(['text'], axis=1)

# Write Dataframe with adjective percent and eventID to csv file
df.to_csv('../data/adjective_percent1.csv')

print("Code executed in - {} min".format((time.time() - start_time)/60))
