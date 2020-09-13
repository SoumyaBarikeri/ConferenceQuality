"""
This file extracts Top Level Domain from website URL and checks if it is trustworthy
"""
import pandas as pd
import numpy as np
import time
# from urlparse import urlparse

start_time = time.time()

# Read conference data downloaded from WikiCfp as Dataframe and set index
df = pd.read_csv('../data/wCfP_data_full_new.csv')
df_reduced = df.reindex(columns=['eventID', 'website'])
df_reduced = df_reduced.set_index('eventID')
# df_reduced['website'] = df_reduced['website'].astype(str)
print(df_reduced.head())


# print(df_reduced.head())
def is_trusted_tld(website):
    """
    Function finds the Top Level Domain from the given website URL

    Parameters
    ----------
    website : str
        Website URL of conference extracted from WikiCfp

    Returns
    -------
    str
        The Top Level Domain
    int
        1 if the Top Level Domain is among the trusted list of TLDs, 0 otherwise

    """
    # List of restrictive and country specific TLDs which can be trusted
    trusted_tld = ['edu', 'mil', 'gov', 'uk', 'it', 'de', 'eu', 'in', 'fr', 'ca', 'cn', 'es', 'pl', 'gr', 'jp', 'br',
                   'au', 'pt', 'ro', 'at', 'id', 'nl', 'ly', 'my', 'tw', 'tr', 'hk', 'sg', 'us']
    website = str(website)
    if website != "nan":
        print(website)

        # get the Top domain in URL
        if 'http://' in website:
            tld = website.split('/')[2].split('.')[-1]
            if 'http://http://' in website:
                tld = website.split('/')[4].split('.')[-1]
        elif 'https://' in website:
            tld = website.split('/')[2].split('.')[-1]
        elif 'http:\\\\' in website:
            print(website)
            tld = website.split('\\')[2].split('.')[-1]
        elif 'https:/' in website:
            tld = website.split('/')[1].split('.')[-1]
        else:
            print(website)
            tld = website.split('/')[2].split('.')[-1]
        print(tld)

        # check if retrieved TLD is in the list of trusted TLDs
        if tld in trusted_tld:
            return tld, 1
        else:
            return tld, 0
    else:
        return 'Missing', np.nan


# Get the TLD for each conference by looping over Dataframe rows and adding the returned values at each row
for index, row in df_reduced.iterrows():
    tld_result = is_trusted_tld(row['website'])
    df_reduced.at[index, 'tld'] = tld_result[0]
    df_reduced.at[index, 'trusted_tld'] = tld_result[1]


# print(df_reduced.tld.value_counts())
# tld_freq = df_reduced.tld.value_counts()
# tld_freq.to_csv('../data/url_tld_freq.csv')

df_reduced.to_csv('../data/url_tld.csv')

print("Code executed in - {}".format(time.time() - start_time))



