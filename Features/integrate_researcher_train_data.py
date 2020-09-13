import pandas as pd
from six import string_types

# get training data
df_data = pd.read_csv('../data/training_data.csv')
# df_committee = pd.read_csv('../data/committee_names_train_data.csv')

# get committee names for all data
df_com = pd.read_csv('../data/CommitteeNames_0_15000.csv')
df1_com = pd.read_csv('../data/CommitteeNames_15000_30000.csv')
df2_com = pd.read_csv('../data/CommitteeNames_30000_45000.csv')
df3_com = pd.read_csv('../data/CommitteeNames_45000_60000.csv')

df_committee = df_com.append(df1_com)
df_committee = df_committee.append(df2_com)
df_committee = df_committee.append(df3_com)

# get researcher related information data
df_researcher = pd.read_csv('../data/researcher_name_data_train.csv')
print(df_researcher.dtypes)
# integrate committe names with train data
df_data = df_data.merge(df_committee, on='eventID')

print(df_data.columns)
print(df_data.shape)

df_data = df_data.reindex(columns=['eventID', 'committeeNames', 'committee_number'])
count = 0
for index, row in df_data.iterrows():
    citations = []
    publications = []
    avg_citation = []
    for name in row['committeeNames'].split('|'):
        print(name)
        researcher = df_researcher[df_researcher['researcher_name'].isin([name])]
        print(researcher.dtypes)
        # for idx, researcher in df_researcher.iterrows():
        #     print(researcher['researcher_name'])
        # if name == researcher['researcher_name']:
        cit = float(researcher['citation_count'])
        publ = researcher['publication_count']
        if "latin-1" in str(publ):
            print('yes')
            publ = 0
        else:
            print('no')
            publ = float(researcher['publication_count'])

        if publ != 0:
            avg_cit = float(cit / publ)
        else:
            avg_cit = 0
        print(cit, publ, avg_cit)
        citations.append(cit)
        publications.append(publ)
        avg_citation.append(avg_cit)
    df_data.at[index, 'total_committee_citation'] = sum(citations)
    df_data.at[index, 'total_committee_publications'] = sum(publications)
    df_data.at[index, 'total_committee_avg_citation'] = sum(avg_citation)
    count = count+1
    print('End of row {}'.format(count))


# df_data = df_data.drop(columns=[''], axis=1)
df_data.to_csv('../data/committee_info_train_data.csv')
