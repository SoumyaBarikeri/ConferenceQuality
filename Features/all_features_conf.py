"""
File generates all the features for a single conference sample
"""
import pandas as pd
import numpy as np

from Features import feature_gen_functions as fgen
from Features import pipeline_functions as pipef


# function to take conference data(as a series) and returns all predictions
def get_all_features(conf_sample):
    """
    Function generates all features for a single instance of conference
    Parameters
    ----------
    conf_sample : pd.Series
        A single conference sample with fields scarped from WikiCfp

    Returns
    -------
    pd.DataFrame
        A DataFrame with single conference with all the generated features

    """
    conf_df = pd.DataFrame(conf_sample)
    conf_df = conf_df.transpose()
    conf_df = conf_df.assign(suspicious_words_found=None)
    conf_df = conf_df.set_index('eventID')
    conf_id = conf_df.index.item()

    # Get citation data of conference series
    print('conf id {}'.format(conf_id))
    df_conf_ser_id = pipef.get_conf_series_id_df(conf_df)
    df_conf_ser_id.to_csv('../data/conf_ser_sample.csv')
    print(df_conf_ser_id.head())
    conference_row = df_conf_ser_id.loc[[conf_id]]
    print(conference_row)
    conf_series_id = conference_row.at[conf_id, 'confSer_adjusted']
    print(conf_series_id)
    conf_df['confSer_adjusted'] = conf_series_id

    conf_df = conf_df.assign(older_conferences=None)
    conf_series = pipef.get_conf_series_citation(conf_id, conf_series_id, df_conf_ser_id)
    print(conf_series[0])
    print(conf_series[1])
    conf_df.at[conf_id, 'older_conferences'] = conf_series[0]
    conf_df['conf_series_citation'] = conf_series[1]

    # Get whois data and process it
    who_is_raw = pipef.get_whois_data_df(conf_df)
    who_is_processed = pipef.get_whois_processed_df(who_is_raw)

    whois_index = who_is_processed.index.item()

    conf_df['private_registration'] = who_is_processed.at[whois_index, 'private_registration']
    conf_df['geo_na_eu'] = who_is_processed.at[whois_index, 'geo_na_eu']
    conf_df['geo_asia'] = who_is_processed.at[whois_index, 'geo_asia']
    conf_df['whois_na_eu'] = who_is_processed.at[whois_index, 'whois_na_eu']
    conf_df['whois_asia'] = who_is_processed.at[whois_index, 'whois_asia']
    conf_df['geo_loc_diff_whois_loc'] = who_is_processed.at[whois_index, 'geo_loc_diff_whois_loc']
    conf_df['website_age'] = who_is_processed.at[whois_index, 'website_age']
    conf_df['identity_hidden'] = who_is_processed.at[whois_index, 'identity_hidden']
    conf_df['completeness'] = who_is_processed.at[whois_index, 'completeness']

    # Get suspicious word count feature
    suspicious_data = fgen.percent_suspicious(conf_df['text'].values[0])
    conf_df['suspicious_words_count'] = suspicious_data[0]
    conf_df['suspicious_words_percentage'] = suspicious_data[1]
    conf_df.at[conf_id, 'suspicious_words_found'] = suspicious_data[2]

    # Get duration between start and submission deadline in days
    duration_days = fgen.get_duration(conf_df['startEvent'].values[0], conf_df['subDL'].values[0])
    conf_df['start_subDl_duration_days'] = duration_days[0]
    conf_df['neg_duration'] = duration_days[1]

    # Get adjective percentage
    adj_freq = fgen.adjective_freq(conf_df['text'].values[0])
    conf_df['adj_percent'] = adj_freq

    # Get trusted TLD
    tld_info = fgen.is_trusted_tld(conf_df['website'].values[0])
    conf_df['tld'] = tld_info[0]
    conf_df['trusted_tld'] = tld_info[1]

    # Get touristic focus feature
    touristic_focus = fgen.has_touristic_focus(conf_df['text'].values[0])
    conf_df['touristic_focus'] = touristic_focus

    # Get information on Program Committee Members
    committee_names = fgen.get_program_committee(conf_df['text'].values[0])
    conf_df['committeeNames'] = committee_names[0]
    conf_df['committee_number'] = committee_names[1]

    researcher_citation = []
    researcher_publication = []
    average_citation = []
    for name in committee_names[0].split('|'):
        researcher_bib = fgen.get_researcher_bib(name)
        researcher_citation.append(researcher_bib[0])
        researcher_publication.append(researcher_bib[1])
        if researcher_bib[1] != 0:
            average_citation.append(researcher_bib[0]/researcher_bib[1])
        else:
            average_citation.append(0)
    conf_df['total_committee_citation'] = sum(researcher_citation)
    conf_df['total_committee_publications'] = sum(researcher_publication)
    conf_df['total_committee_avg_citation'] = sum(average_citation)

    if committee_names[1] != 0:
        conf_df['avg_cit_per_person'] = sum(average_citation)/committee_names[1]
    else:
        conf_df['avg_cit_per_person'] = 0

    return conf_df
