"""
This file generates a unique list of researchers and then extracts researcher data from Dimensions
"""
import pandas as pd
from dimcli.shortcuts import dslquery_json as dslquery
from Data_Acquisition.queryDimensions import get_publications_from_researcher_name
import time

start_time = time.time()

df = pd.read_csv('../data/CommitteeNames_0_15000.csv')
df1 = pd.read_csv('../data/CommitteeNames_15000_30000.csv')
df2 = pd.read_csv('../data/CommitteeNames_30000_45000.csv')
df3 = pd.read_csv('../data/CommitteeNames_45000_60000.csv')

df_total = df.append(df1)
df_total = df_total.append(df2)
df_total = df_total.append(df3)


print(df_total.head())

# Generate unique researcher list
count = 1
names_list_unique = []

for index, row in df_total.iterrows():
    for name in row['committeeNames'].split('|'):
        count = count + 1
        if name not in names_list_unique:
            names_list_unique.append(name)

print(count)
print(len(names_list_unique))
author_data = pd.DataFrame(names_list_unique, columns=['researcher_name'])
author_data.to_csv('../data/CommitteeNames_full_list.csv')

names_list_unique = pd.read_csv('../data/CommitteeNames_full_list.csv', encoding='utf-8')
print(len(names_list_unique))


citation_list = []
publ_number_list = []

# List of Computer Science fields
search_string = ['artificial', 'computer', 'computing', 'systems', 'information', 'data', 'networks', 'security', 'computational',
                 'network', 'dataset', 'framework', 'internet', 'cloud', 'database', 'IoT', 'software', 'server', 'hardware',
                 'robot', 'programming', 'blockchain', 'CPU', 'ontology', 'Semantic Web', 'developers', 'API', 'Web', 'semantics',
                 'technology', 'system', 'machine translation', 'natural language processing', 'tool', 'processing system',
                 'language technology', 'processing system', 'net', 'machine learning', 'classification', 'algorithm', 'model',
                 'application', 'automation', 'modules', 'NLP', 'resources', 'recognition', 'corpus', 'prototype', 'interface',
                 'modeling', 'software engineering', 'code', 'processors', 'chip']


for index, row in names_list_unique.iterrows():
    name = row['researcher_name']
    name = name.replace(' �', '')
    name = name.replace('�', '')
    skip = 0
    pubs = []
    total_pubs = []
    publ_number = 0
    try:
        while (skip == 0) or (len(pubs) == 1000):
            pubs = dslquery(get_publications_from_researcher_name(name, skip=skip)).get('publications', [])
            total_pubs += pubs
            skip += 1000  # Provide an offset of 1000 as only 1000 publications are returned at a time
        if len(total_pubs) != 0:
            # Get citations total for only computer science related publications
            publ_number = len(total_pubs)
            citations = sum(x['times_cited'] for x in total_pubs if len(x) > 1 if
                            len(set(x['concepts']).intersection(search_string)) > 0)
        else:
            citations = 0
    except Exception as error:
        print('Author citation retrieval not possible for {}'.format(name))
        publ_number = error
        citations = 0
    print('Researcher {} - citation {}, publication - {}'.format(name, citations, publ_number))
    names_list_unique.at[index, 'citation_count'] = citations
    names_list_unique.at[index, 'publication_count'] = publ_number


names_list_unique.to_csv('../data/researcher_data.csv')

print("Code executed in - {}".format(time.time() - start_time))
