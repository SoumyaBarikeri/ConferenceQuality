"""
This file extracts Program committee member names from conference Call for Paper
"""
from Features.programCommitteeStandfordNer import getProgramCommittee
import time
import pandas as pd

start_time = time.time()

# Read conference data downloaded from WikiCfp as Dataframe and set index
df = pd.read_csv('../data/training_data.csv')
df_reduced = df.reindex(columns=['eventID', 'text'])
df_reduced = df_reduced.set_index('eventID')

committee_names = []
author_citation_list = []

# Get author names for each conference by looping over rows of Dataframe and assign to new columns in Dataframe
for index, row in df_reduced.iterrows():
    if row['text'] != "[Empty]" and row['text'] != "":
        text_cfp = row['text']
        text_cfp = text_cfp.replace('/', ' ')
        text_cfp = text_cfp.replace('\n ', ',')
        print('For conf ID - {}'.format(index))
        names = getProgramCommittee(text_cfp)
        if len(names) > 0:
            names = "|".join(names)
            df_reduced.loc[index, 'committeeNames'] = names
            df_reduced.loc[index, 'committee_number'] = len(names.split('|'))
        else:
            df_reduced.loc[index, 'committeeNames'] = "Names missing"
            df_reduced.loc[index, 'committee_number'] = 0
    else:
        df_reduced.loc[index, 'committeeNames'] = "Data not valid"
        df_reduced.loc[index, 'committee_number'] = 0


df_reduced = df_reduced.reset_index()
df_reduced = df_reduced.drop(['text'], axis=1)

# Write Dataframe with Program Committee information and eventID to csv
df_reduced.to_csv('../data/committee_names_train_data.csv', index=False)

print("Code executed in - {}".format(time.time() - start_time))
