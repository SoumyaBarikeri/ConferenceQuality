"""
This file generates features regarding suspicious words in Call for Paper
"""
import pandas as pd

# Read conference data downloaded from WikiCfp as Dataframe and set index
df = pd.read_csv('../data/wCfP_data_full_new.csv')
df_reduced = df.reindex(columns=['eventID', 'text', 'title'])
df_reduced = df_reduced.set_index('eventID')
df_reduced = df_reduced.assign(suspicious_words_count=None)
df_reduced = df_reduced.assign(suspicious_words_percentage=None)
df_reduced = df_reduced.assign(suspicious_words_found=None)

suspicious_words = pd.read_csv('../data/suspicious_words_pred_publs.csv', header=0)
suspicious_words = list(suspicious_words['words'])


def percent_suspicious(cfp_text):
    """
    Function identifies if predefined suspicious words are found in Conference text

    Parameters
    ----------
    cfp_text : str
        Conference text from WikiCfp

    Returns
    -------
    int
        Number of suspicious words found in text
    float
        Percentage of suspicious words found in text
    list
        The list of suspicious words found in text

    """
    count = 0
    words_found = []
    for phrase in suspicious_words:
        if phrase in cfp_text:
            count = count+1
            words_found.append(phrase)

    if not words_found:
        words_found = 'No words found'
    return count, (count/len(cfp_text.split(' ')))*100, words_found


# Get suspicious words feature for all conferences by looping over rows of Dataframe
for index, row in df_reduced.iterrows():
    suspicious = percent_suspicious(row['text'])
    df_reduced.at[index, 'suspicious_words_count'] = suspicious[0]
    df_reduced.at[index, 'suspicious_words_percentage'] = suspicious[1]
    df_reduced.at[index, 'suspicious_words_found'] = suspicious[2]


df_reduced = df_reduced.drop(columns=['text', 'title'], axis=1)
df_reduced.to_csv('../data/suspicious_words.csv')
