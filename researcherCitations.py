"""
This file extracts researcher data for each conference by looping over the conference DataFrame
"""
from dimcli.shortcuts import dslquery_json as dslquery
import time
import pandas as pd
from Data_Acquisition.queryDimensions import get_publications_from_researcher_name

start_time = time.time()

# Read conference data downloaded from WikiCfp as Dataframe and set index
df = pd.read_csv('../data/CommitteeNames_1000.csv')
df.set_index('eventID')
df['committeeNames_encoded'] = map(lambda x: x.encode('utf-8'), df['committeeNames'])
df = df.assign(researcher_citation=None)
df = df.assign(researcher_publication=None)

name_list = []
search_string = ['artificial', 'computer', 'computing', 'systems', 'information', 'data', 'networks', 'security', 'computational',
                 'network', 'dataset', 'framework', 'internet', 'cloud', 'database', 'IoT', 'software', 'server', 'hardware',
                 'robot', 'programming', 'blockchain', 'CPU', 'ontology', 'Semantic Web', 'developers', 'API', 'Web', 'semantics',
                 'technology', 'system', 'machine translation', 'natural language processing', 'tool', 'processing system',
                 'language technology', 'processing system', 'net', 'machine learning', 'classification', 'algorithm', 'model',
                 'application', 'automation', 'modules', 'NLP', 'resources', 'recognition', 'corpus', 'prototype', 'interface',
                 'modeling', 'software engineering', 'code', 'processors', 'chip']

for index, row in df.iterrows():
    print(row['committeeNames'])
    researcher_citation_for_conf = []
    researcher_publ_for_conf = []
    publ_number = 0
    if row['committeeNames'] != 'Names missing' and row['committeeNames'] != 'Data not valid':
        row_list = row['committeeNames'].split('|')
        for name in list(set(row_list)):
            skip = 0
            pubs = []
            total_pubs = []
            try:
                while (skip == 0) or (len(pubs) == 1000):
                    pubs = dslquery(get_publications_from_researcher_name(name, skip=skip)).get('publications', [])
                    total_pubs += pubs
                    skip += 1000
            except UnicodeEncodeError:
                print('Author citation retrieval not possible')
            if len(total_pubs) != 0:
                # Get citations for only computer science related publications
                publ_number = len(total_pubs)
                citations = sum(x['times_cited'] for x in total_pubs if len(x) > 1 if
                                len(set(x['concepts']).intersection(search_string)) > 0)
            else:
                citations = 0
            print('Citation for researcher {} is {}'.format(name, citations))
            researcher_citation_for_conf.append(citations)
            researcher_publ_for_conf.append(publ_number)
    else:
        # publ_number = 0
        citations = None # Conferences without any mention of Program committee
        researcher_citation_for_conf.append(citations)
        researcher_publ_for_conf.append(publ_number)
    print(researcher_citation_for_conf)
    df.at[index, 'researcher_citation'] = researcher_citation_for_conf
    df.at[index, 'researcher_publication'] = researcher_publ_for_conf


df = df.drop(['committeeNames_encoded'], axis=1)
df.to_csv('/Users/soumya/PycharmProjects/QualityConferences/data/researcher_citation.csv', index=False)


print("Code executed in - {}".format(time.time() - start_time))

