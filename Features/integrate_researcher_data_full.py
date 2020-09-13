"""
This file integrates Program committee of each conference with their corresponding bibliometrics
"""
import pandas as pd
import time
from six import string_types


start_time = time.time()

# get committee names for all data
df_com = pd.read_csv('../data/CommitteeNames_0_15000.csv')
df1_com = pd.read_csv('../data/CommitteeNames_15000_30000.csv')
df2_com = pd.read_csv('../data/CommitteeNames_30000_45000.csv')
df3_com = pd.read_csv('../data/CommitteeNames_45000_60000.csv')

df_committee = df_com.append(df1_com)
df_committee = df_committee.append(df2_com)
df_committee = df_committee.append(df3_com)

# get researcher related data
df_researcher = pd.read_csv('../data/researcher_data_2.csv', index_col='researcher_name')

print(df_researcher.dtypes)
print(df_researcher.head())

print(df_committee.columns)
print(df_committee.shape)
print(df_committee.head())

df_committee = df_committee.set_index('eventID')

count = 0
# Integrate Program committee of each conference with citation data
for index, row in df_committee.iterrows():
    citations = []
    publications = []
    avg_citation = []
    for name in row['committeeNames'].split('|'):
        if name:
            # For the Program committee name in conference, find the row in researcher data that matches the name
            researcher = df_researcher.loc[name]
            print(researcher)
            cit = researcher['citation_count']
            publ = researcher['publication_count']
            if "latin-1"in str(publ):
                publ = 0
            elif "Connection aborted" in str(publ):
                publ = 0
            else:
                publ = float(researcher['publication_count'])

            # Calculate average citation per publication
            if publ != 0:
                avg_cit = float(cit / publ)
            else:
                avg_cit = 0
            print(cit, publ, avg_cit)
            citations.append(cit)
            publications.append(publ)
            avg_citation.append(avg_cit)

    # Append the sum of citation data of all Program committee members in the conference
    df_committee.at[index, 'total_committee_citation'] = sum(citations)
    df_committee.at[index, 'total_committee_publications'] = sum(publications)
    df_committee.at[index, 'total_committee_avg_citation'] = sum(avg_citation)
    count = count+1
    print('End of row {}'.format(count))


df_committee.to_csv('../data/committee_info_full.csv')

print("Code executed in - {}".format(time.time() - start_time))

