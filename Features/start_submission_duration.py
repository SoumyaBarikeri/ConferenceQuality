"""
This file calculates the duration between Start of conference and paper Submission deadline for conferences
"""
import pandas as pd
import numpy as np
import time

start_time = time.time()

# Read conference data downloaded from WikiCfp as Dataframe and set index
df = pd.read_csv('../data/wCfP_data_full_new.csv')
df = df.reindex(columns=['eventID', 'startEvent', 'subDL'])
df = df.set_index('eventID')

# Convert dates into datetime datatype by specifying the format
df['startEvent'] = pd.to_datetime(df['startEvent'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
df['subDL'] = pd.to_datetime(df['subDL'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')

print(df['subDL'].isnull().sum())
print(df['startEvent'].isnull().sum())


def get_duration(start_date, submission_date):
    """
    Function calculates the duration between Start of conference and its submission deadline in days

    Parameters
    ----------
    start_date : datetime
        Conference start date
    submission_date : datetime
        Paper submission deadline

    Returns
    -------
    duration : int
        Duration  between start and submission dates in days
    negative_duration : int
        Value is 1 if duration is negative

    """
    if start_date is pd.NaT or submission_date is pd.NaT:
        return np.nan, np.nan
    else:
        duration = (start_date - submission_date).days
        duration = int(duration)
        if duration < 0:
            return duration, 1
        else:
            return duration, 0


# Get duration information for each conference by looping over rows in Dataframe and assign them to new column
for index, row in df.iterrows():
    start_sub_duration = get_duration(row['startEvent'], row['subDL'])
    df.at[index, 'start_subDl_duration_days'] = start_sub_duration[0]
    df.at[index, 'neg_duration'] = start_sub_duration[1]


df = df.drop(columns=['startEvent', 'subDL'], axis=1)

# Write Dataframe with eventID and duration information to CSV
df.to_csv('../data/duration_days.csv')
