import pandas as pd
import datetime


import some_standard_funcs as ssf
import some_text_stuff as sts


# read data and join on eventID
df = ssf.standard_data_import()
df = ssf.remove_duplicates(df)
df = ssf.remove_journals(df)
df = df.reindex(columns=['title_changed', 'title', 'iteration', 'startEvent', 'year'])

df_indexed = pd.read_csv('data/wCfP_titles_indexed.csv', index_col='eventID')
df_indexed = df.join(df_indexed)

# initilize dicts and df for summary of results
dict_matches_found = {}
dict_first_level_matches_found = {}
dict_wCfP_confs = {}
dict_dim_confs = {}
df_result = pd.DataFrame()


# load and adjust dimensions data
def dateparse(val):
    try:
        return pd.datetime.strptime(val, '%Y-%m-%d').date()
    except:
        return pd.NaT


df_dim = pd.read_csv('data/dim_titles_indexed_full.csv',
                     parse_dates=['date'], date_parser=dateparse)

# print(df_dim[df_dim.year == 0])
# print(df[df.year == 0])

# print(df_dim)
eventID_tbc = 68642
dim_id_tbc = 4363

# iterate over all years available
for year in range(2007, 2020):
    year_str = str(year)
    dict_matches_found[year_str] = 0
    dict_first_level_matches_found[year_str] = 0
    df_idx_red = df_indexed[df_indexed.year == year]
    # print(df_idx_red.at[eventID_tbc, 'title'])

    df_dim_red = df_dim[(df_dim.date <= pd.Timestamp(year + 1, 6, 30, 0)) &
                        (df_dim.date >= pd.Timestamp(year, 1, 1, 0))]
    # print(df_dim_red.at[dim_id_tbc, 'title'])

    # fill result dicts
    dict_wCfP_confs[year_str] = df_idx_red.shape[0]
    dict_dim_confs[year_str] = df_dim_red.shape[0]


    # go through wCfP data, find possible matches and make comparisons

    for idx in df_idx_red.index:
        # print('\nNew Conference:')
        # print(row['title_changed'])
        # if title tokens is empty (no non-stopwords skip matching and comparison)
        if not isinstance(df_idx_red.at[idx, 'title_tokens'], str):
            # if idx == eventID_tbc:
            #     print(df_idx_red.at[idx, 'title_tokens'])
            #     print('title token was not a string')
            continue

        # if idx == eventID_tbc:
        #     print('\neventID being checked for matches')
        matches = ssf.find_indexed_title(df_idx_red.at[idx, 'title_tokens'].split(), df_dim_red)
        if matches:
            # print('first level match found')
            # if idx == eventID_tbc:
            #     print('\nsome matches found:')
            #     print(matches)
            dict_first_level_matches_found[year_str] += 1
            if not isinstance(df_idx_red.at[idx, 'title_changed'], str):
                break

            indexed_title_changed = df_idx_red.at[idx, 'title_changed']
            best_result, best_sim, best_match = False, 0, None
            for match in matches:
                # if idx == eventID_tbc and match[0] == dim_id_tbc:
                #     print('\nconferences were matched')
                dim_match = df_dim.loc[match[0]]
                if dim_match['year'] != 0 and df_idx_red.at[idx, 'year'] != 0:
                    if dim_match['year'] != df_idx_red.at[idx, 'year']:
                        continue
                    if dim_match['iteration'] != 0 and df_idx_red.at[idx, 'iteration'] != 0:
                        if dim_match['iteration'] != df_idx_red.at[idx, 'iteration']:
                            continue
                elif dim_match['date'] <= pd.Timestamp(df_idx_red.at[idx, 'startEvent']):
                    continue

                # if idx == eventID_tbc and match[0] == dim_id_tbc:
                #     print('\nyears or dates or iterations were matched')

                result, sim = ssf.compare_titles(indexed_title_changed, dim_match['title_changed'],
                                                 threshold=0.7)
                # if result:
                #     print('second level match found')

                if sim > best_sim:
                    best_result = result
                    best_sim = sim
                    best_match = dim_match
                # ToDo make sure that best match is taken, also check why that thing with the date still worked (are there some missing?) --> should be done
                # also that date stuff is mainly because the publication dates from dim are somewhat questionable (probably they refer to the
                # conference dates or are just very rough estimates)
                # if idx == eventID_tbc and match[0] == dim_id_tbc:
                #     print('similarity: %f' % sim)

            if best_result:
                # if idx == eventID_tbc and best_match[0] == dim_id_tbc:
                #     print('given ID combination was the best match')
                dict_matches_found[year_str] += 1
                df_result = df_result.append(pd.Series({'title_dim_changed': best_match['title_changed'], 'id_dim': best_match.name,
                                                        'title_dim': best_match['title'],
                                                        'iteration_dim': best_match['iteration'], 'date_dim': best_match['date'],
                                                        'title_wCfP_changed': df_idx_red.at[idx, 'title_changed'], 'title_wCfP': df_idx_red.at[idx, 'title'],
                                                        'iteration_wCfP': df_idx_red.at[idx, 'iteration'], 'date_wCfP': df_idx_red.at[idx, 'startEvent'],
                                                        'similarity': best_sim, 'citations': best_match['citations'], 'fcr': best_match['fcr'],
                                                        'papers': best_match['papers'], 'eventID': idx}, name=idx))

                    # print('Match found')
                    # break

print(df_result)
df_result.set_index('eventID')
# print(df_result)


# print results
for (key1, value1), (key2, value2), (key3, value3), (key4, value4) in \
        zip(dict_wCfP_confs.items(), dict_dim_confs.items(), dict_first_level_matches_found.items(), dict_matches_found.items()):
    print('\n' + key1 + ':')
    print('wCfP: %i' % value1)
    print('dim: %i' % value2)
    print('First Level Matches: %i' % value3)
    print('Matches: %i' % value4)
    # print('Part of wCfP matched: %f' % value3/value1)
    # print('Part of dim matched: %f' % value3/value2)

df_result.to_csv('data/wCfP_with_dim_citations_full_3.csv', index=True)
