import pandas as pd
import numpy as np
import time


import some_standard_funcs as ssf


# import and prepare data
df = ssf.standard_data_import(True)
df = ssf.remove_duplicates(df)
df = ssf.remove_journals(df)
df_indexed = pd.read_csv('../data/wCfP_titles_indexed.csv', index_col='eventID')
df = df_indexed.join(df)
df = df.dropna(subset=['title_tokens'])

# determine number of conference series coming from wikiCfP
df_temp = df.dropna(subset=['confSer'])
series = df_temp['confSer'].unique().tolist()
series_count = len(series)

df['confSer_adjusted'] = np.nan
df['confSer_adjusted_sim'] = np.nan

# assign conference Series indices to conferences that come with a conference Series given from wikiCfP
for idx in df.index:
    if not pd.isna(df.at[idx, 'confSer']):
        df.at[idx, 'confSer_adjusted'] = series.index(df.at[idx, 'confSer'])

start = 2008
end = 2019

match_count = 0

start_time = time.time()

# iterate over all given years and compare all conferences from one year to all conferences of the following years
for year in range(start, end):
    print(year)
    df_year = df[df.year == year]

    # iterate over all conferences in a year
    for idx in df_year.index:
        if idx % 50 == 0:
            print(idx)
        if not isinstance(df_year.at[idx, 'title_changed'], str):
            continue
        if not pd.isna(df_year.at[idx, 'confSer_adjusted']):
            # if conference is part of a series save that series as a DataFrame
            df_ser = df[df.confSer_adjusted == df_year.at[idx, 'confSer_adjusted']]

        # iterate over all following years
        for year_comp in range(year+1, end+1):
            # check if there is already a conference for the given year in the conference series, if so skip the year
            if not pd.isna(df_year.at[idx, 'confSer_adjusted']):
                if year_comp in df_ser['year']:
                    break

            df_comp = df[df.year == year_comp]
            best_result, best_sim, best_match = False, 0, None
            matches = ssf.find_indexed_title(df_year.at[idx, 'title_tokens'].split(), df_comp, threshold=0.75)

            # if matches were found iterate over them to make detailed comparison
            if matches:
                for match in matches:
                    comp_match = df_comp.loc[match[0]]
                    # if iterations exist, check if iterations fit
                    if comp_match['iteration'] != 0 and df_year.at[idx, 'iteration'] != 0:
                        if comp_match['iteration'] <= df_year.at[idx, 'iteration']:
                            continue

                    # make levensthein comparison and save result if the result is the best fit so far
                    result, sim = ssf.compare_titles(df_year.at[idx, 'title_changed'], comp_match['title_changed'],
                                                     threshold=0.95)
                    if result:
                        print('levenshtein match found')
                        if sim > best_sim:
                            best_result = True
                            best_sim = sim
                            best_match = match

                # if there is a match, take it and mark both conferences with the series
                if best_match:
                    match_count += 1
                    if pd.isna(df_year.at[idx, 'confSer_adjusted']):
                        # if conference is not yet part of a series, check if matched conference has one
                        if pd.isna(df.at[best_match[0], 'confSer_adjusted']):
                            # if matched conference has no series, assing both conferences to a new series
                            df.at[idx, 'confSer_adjusted'] = series_count
                            df.at[idx, 'confSer_adjusted_sim'] = best_sim
                            df.at[best_match[0], 'confSer_adjusted'] = series_count
                            df.at[best_match[0], 'confSer_adjusted_sim'] = best_sim
                            series_count += 1
                        else:
                            # if matched conference has a series, assign conference to it
                            df.at[idx, 'confSer_adjusted'] = df.at[best_match[0], 'confSer_adjusted']
                            df.at[idx, 'confSer_adjusted_sim'] = best_sim
                    else:
                        if pd.isna(df.at[best_match[0], 'confSer_adjusted']):
                            # if conference is part of a series and matched conference is not part of one, assign
                            # the match to the series of the conference
                            df.at[best_match[0], 'confSer_adjusted'] = df.at[idx, 'confSer_adjusted']
                            df.at[best_match[0], 'confSer_adjusted_sim'] = best_sim

print('time elapsed:')
print(time.time() - start_time)

print('matches: %i' % match_count)


df = df.reindex(columns=['confSer_adjusted', 'confSer_adjusted_sim'])
df.to_csv('../data/confSeries.csv', index=True)

'''
for i, idx in enumerate(df.index):
    year = 
    df_temp = df
    matches = ssf.find_indexed_title(df.at[idx, 'title_tokens'], df.iloc[i+1:])
'''
