"""
This file gets detects predatory conference and predicts quality of conferences.
Basically a pipeline for backend of an application that can be used by academic/researcher
"""
import pandas as pd
import numpy as np
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from Features.feature_gen_functions import get_topics_conf
from Features.all_features_conf import get_all_features
from joblib import load
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


# Read conferences scraped from wikiCfp
df = pd.read_csv('../data/wCfP_data_full_new_3.csv')

# Pass the conference that is to be checked
conf_row = df.iloc[7]
print(conf_row)

# get CSO topics for the conference
conf_topics = get_topics_conf(conf_row)

# Apply CS/Non-CS classifier model
filename = '../models/modelRnn_wCfp_full_390.h5'
model = load_model(filename, compile=False)
print('\nmodel_loaded\n')

X = conf_topics[0] + conf_topics[1]
print(X)
X = str(X)

# Convert input words to vectors
max_words = 50000
max_len = 250
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts([X])
test_sequences = tok.texts_to_sequences([X])
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)


y_pred = model.predict(np.array(test_sequences_matrix))
print(y_pred)

if y_pred > 0.5:
    print('Conference is classified as Computer science, proceeding further')

    # Generate sensible features for Machine learning algorithms
    conf_row_features = get_all_features(conf_row)
    print(conf_row_features)

    # Apply pre-processing
    conf_row_features['trusted_tld'] = conf_row_features['trusted_tld'].replace(np.nan, 0)
    conf_row_features['neg_duration'] = conf_row_features['neg_duration'].replace(np.nan, 0)
    conf_row_features['start_subDl_duration_days'] = conf_row_features['start_subDl_duration_days'].replace(np.nan,
                                                                                                            conf_row_features[
                                                                                                                'start_subDl_duration_days'].mean())
    conf_row_features['touristic_focus'] = conf_row_features['touristic_focus'].replace(np.nan, 0)
    conf_row_features['private_registration'] = conf_row_features['private_registration'].replace(np.nan, 0)
    conf_row_features['identity_hidden'] = conf_row_features['identity_hidden'].replace(np.nan, 0)
    conf_row_features['completeness'] = conf_row_features['completeness'].replace(np.nan, 0)
    conf_row_features['conf_series_citation'] = conf_row_features['conf_series_citation'].replace(np.nan, 0)
    conf_row_features['touristic_focus'] = conf_row_features['touristic_focus'].replace(np.nan, 0)
    conf_row_features['geo_loc_diff_whois_loc'] = conf_row_features['geo_loc_diff_whois_loc'].replace("True", 1)
    conf_row_features['geo_loc_diff_whois_loc'] = conf_row_features['geo_loc_diff_whois_loc'].replace("False", 0)

    if conf_row_features['committee_number'].values[0] != 0:
        conf_row_features['avg_cit_per_person'] = conf_row_features['total_committee_avg_citation'] / conf_row_features[
            'committee_number']
    else:
        conf_row_features['avg_cit_per_person'] = 0

    # Apply model 1 - Identify conference as Predatory or not
    conf_row_features = conf_row_features.reindex(
        columns=['touristic_focus', 'text_length', 'trusted_tld', 'adj_percent', 'suspicious_words_count',
                 'start_subDl_duration_days', 'neg_duration', 'committee_number', 'total_committee_citation',
                 'total_committee_publications', 'total_committee_avg_citation', 'avg_cit_per_person',
                 'conf_series_citation', 'private_registration', 'identity_hidden', 'completeness',
                 'geo_na_eu', 'geo_asia', 'whois_na_eu', 'whois_asia', 'geo_loc_diff_whois_loc',
                 'website_age'])

    conf_row_features = conf_row_features.iloc[0]
    conf_row_features = np.array(conf_row_features)
    conf_row_features = conf_row_features.reshape(1, -1)
    model_level_1 = load('../models/classifier_GB.joblib')
    predatory = model_level_1.predict(conf_row_features)

    # Apply model 2 - Predict conference quality
    if predatory == 1:
        print('Conference is predatory, exiting')
    else:
        print('Conference is non-predatory, predicting its quality')
        model_level_2 = load('../models/regressor_RF.joblib')
        conf_quality = model_level_2.predict(conf_row_features)
        print('The quality of conference is {}'.format(conf_quality))

else:
    print('Conference is classified as Non Computer science, exiting')





