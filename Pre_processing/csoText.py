"""
This file fetches conference topics using Computer Science Ontology Classifier
"""
import classifier.classifier as CSO
import json
from nltk.stem import PorterStemmer
import pandas as pd


def stemwords(all_words):
    """
    Function to get stem of all conference topics using Porter stemmer algorithm

    Parameters
    ----------
    all_words
        List of all topics

    Returns
    -------
    output
        Stemmed form of the topics

    """
    st = PorterStemmer()
    text = all_words

    output = []
    for sentence in text:
        output.append(" ".join([st.stem(i) for i in sentence.split()]))

    return output


def getTopicsConf(conference):
    """
    Function gets Semantic, Syntactic and Enhanced topics of Conference by running CSO classifier on Conference title
    and text

    Parameters
    ----------
    conference
        Conference data i.e. title and text
    df2
        Dataframe with manual labels for Computer Science/Non Computer Science conferences
    i
        Iterator over each conference

    Returns
    -------
    topics
        The syntactic, semantic and enhanced topics that are returned by CSO classifier

    """
    eventID = conference['eventID']
    title = conference['title']
    print('title is' + title)
    text = conference['text']
    json_string = json.dumps({'key1': title, 'key2': text})
    result = CSO.run_cso_classifier(json_string, modules="both", enhancement="first")
    print(result['semantic'])
    semantic_topics = "|".join(result['semantic'])
    syntactic_topics = "|".join(result['syntactic'])
    enhanced_topics = "|".join(result['enhanced'])
    return semantic_topics, syntactic_topics, enhanced_topics


def getTopics(conference, df2, i):
    """
    Function gets Semantic, Syntactic and Enhanced topics of Conference by running CSO classifier on Conference title
    and text, adds the topics to the Dataframe as new columns

    Parameters
    ----------
    conference
        Conference data i.e. title and text
    df2
        Dataframe with manual labels for Computer Science/Non Computer Science conferences
    i
        Iterator over each conference

    Returns
    -------
    topics
        The syntactic and semantic topics returned by CSO classifier

    """
    eventID = df2.index[i]
    title = conference[0]
    print('title is' + title)
    text = conference[1]
    json_string = json.dumps({'key1': title, 'key2': text})
    result = CSO.run_cso_classifier(json_string, modules="both", enhancement="first")
    topics = result['semantic'] + result['syntactic']
    print(result['semantic'])
    semantic_topics = "|".join(result['semantic'])
    syntactic_topics = "|".join(result['syntactic'])
    enhanced_topics = "|".join(result['enhanced'])
    df2.ix[eventID, 'semantic'] = semantic_topics
    df2.ix[eventID, 'syntactic'] = syntactic_topics
    df2.ix[eventID, 'enhanced'] = enhanced_topics

    return topics


df = pd.read_csv('data/wCfP_data_full_new.csv')
df_Labeled = pd.read_csv('data/wCfP_cs_manual_labels_extended_420.csv', delimiter=';')

df_reduced = df.loc[:, ['eventID', 'title', 'text']]
# Merge WikiCfp data with conferences manually labeled as Computer science or Non Computer science
df_Labeled = pd.merge(df_Labeled, df_reduced, on='eventID')

columns = ['eventID', 'title', 'text','csLabel']
df2 = pd.DataFrame(df_Labeled, columns=columns)
# Create new columns in Dataframe for topics
df2['semantic'] = ""
df2['syntactic'] = ""
df2['enhanced'] = ""
df2['semantic'] = df2['semantic'].astype(object)
df2['syntactic'] = df2['syntactic'].astype(object)
df2['enhanced'] = df2['enhanced'].astype(object)

df2.set_index('eventID')

result = [getTopics(conf, df2, i) for i, conf in zip(range(df2.shape[0]), df2[['title', 'text']].values)]


all_topics = []
for row in result:
    for word in row:
        all_topics.append(word)

# Get a count of all distinct topics
wordCount = [[x, all_topics.count(x)] for x in set(all_topics)]

df2 = df2.drop(['title', 'text'], axis =1)

# Write the conference topics to csv file
df2.to_csv('data/conftopics.csv')


