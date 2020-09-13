from model_building import model_training_evaluation as mte
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
import xgboost as xgb
from sklearn.svm import SVR
import numpy as np
import collections
import some_standard_funcs as ssf
import time
from joblib import dump
import warnings

import statsmodels.api as sm


start_time = time.time()

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


data = ssf.standard_data_import(only_cs=True)
data = data.reindex(columns=['text_length'])

# adding labels
df_pred_labels = pd.read_csv('../data/predatory_conferences_2.csv', sep=';', index_col='eventID')
df_pred_labels = df_pred_labels.drop(['way_of_identification'], axis=1)
df_good_labels = pd.read_csv('../data/wCfP_integrated_dim_citations.csv', index_col='eventID')
df_good_labels = df_good_labels[(df_good_labels.similarity >= 0.95) & (df_good_labels.citations > 10)]
df_good_labels = df_good_labels.reindex(columns=['citations', 'fcr', 'papers'])

data = data.join(df_pred_labels)
data = data.join(df_good_labels)

# predatory column comes from df_pred_labels and only positive ones must be added
for idx in data.index:
    if np.isnan(data.at[idx, 'predatory']):
        if data.at[idx, 'citations'] >= 10:
            data.at[idx, 'predatory'] = 0

# adding features
df_touristic = pd.read_csv('../data/touristic_focus.csv', index_col='eventID')
df_tld = pd.read_csv('../data/url_tld.csv', index_col='eventID')
df_adjective = pd.read_csv('../data/adjective_percent1.csv', index_col='eventID')
df_duration = pd.read_csv('../data/duration_days.csv', index_col='eventID')
df_suspicious = pd.read_csv('../data/suspicious_words.csv', index_col='eventID')
df_committee = pd.read_csv('../data/committee_info_full.csv', index_col='eventID')
df_conf_series = pd.read_csv('../data/confSeries_citation.csv', index_col='eventID')
df_who_is_info = pd.read_csv('../data/whois_processed.csv', index_col='eventID')


data = data.join(df_touristic)
data = data.join(df_tld)
data = data.join(df_adjective)
data = data.join(df_duration)
data = data.join(df_suspicious)
data = data.join(df_committee)
data = data.join(df_conf_series)
data = data.join(df_who_is_info)

print(data.columns)

# Handling missing values
data['tld'] = data['tld'].replace(np.nan, '')
data['trusted_tld'] = data['trusted_tld'].replace(np.nan, 0)
data['neg_duration'] = data['neg_duration'].replace(np.nan, 0)
data['start_subDl_duration_days'] = data['start_subDl_duration_days'].replace(np.nan, data['start_subDl_duration_days'].mean())
data['touristic_focus'] = data['touristic_focus'].replace(np.nan, 0)
data['private_registration'] = data['private_registration'].replace(np.nan, 0)
data['identity_hidden'] = data['identity_hidden'].replace(np.nan, 0)
data['completeness'] = data['completeness'].replace(np.nan, 0)
data['conf_series_citation'] = data['conf_series_citation'].replace(np.nan, 0)
data['geo_loc_diff_whois_loc'] = data['geo_loc_diff_whois_loc'].replace("True", 1)
data['geo_loc_diff_whois_loc'] = data['geo_loc_diff_whois_loc'].replace("False", 0)

# Generate feature average citation count per Program committee member
for index, row in data.iterrows():
    if row['committee_number'] != 0:
        data.at[index, 'avg_cit_per_person'] = row['total_committee_avg_citation']/row['committee_number']
    else:
        data.at[index, 'avg_cit_per_person'] = 0


# test data must be split from full data based on labeling
train = data.dropna(subset=['predatory'])
test = data[pd.isnull(data['predatory'])]

train = train.astype({'predatory': 'int32'})
print(train.head())
print('Training examples: %i' % train.shape[0])
print('Testing examples: %i' % test.shape[0])
print('Missing values in train data')
print(train.isna().sum())



# Uncomment below code to normalise train data
'''
train['text_length'] = (train['text_length']-train['text_length'].mean())/train['text_length'].std()
train['start_subDl_duration_days'] = (train['start_subDl_duration_days']-train['start_subDl_duration_days'].mean())/train['start_subDl_duration_days'].std()
train['suspicious_words_count'] = (train['suspicious_words_count']-train['suspicious_words_count'].mean())/train['suspicious_words_count'].std()
train['total_committee_citation'] = (train['total_committee_citation']-train['total_committee_citation'].mean())/train['total_committee_citation'].std()
train['total_committee_publications'] = (train['total_committee_publications']-train['total_committee_publications'].mean())/train['total_committee_publications'].std()
train['total_committee_avg_citation'] = (train['total_committee_avg_citation']-train['total_committee_avg_citation'].mean())/train['total_committee_avg_citation'].std()
train['avg_cit_per_person'] = (train['avg_cit_per_person']-train['avg_cit_per_person'].mean())/train['avg_cit_per_person'].std()
train['conf_series_citation'] = (train['conf_series_citation']-train['conf_series_citation'].mean())/train['conf_series_citation'].std()
train['website_age'] = (train['website_age']-train['website_age'].mean())/train['website_age'].std()
'''

# Get features and target variable from train data
train_features, train_target = train.reindex(columns=['touristic_focus', 'text_length', 'trusted_tld', 'adj_percent', 'suspicious_words_count',
                                                      'start_subDl_duration_days', 'neg_duration', 'committee_number', 'total_committee_citation',
                                                      'total_committee_publications', 'total_committee_avg_citation', 'avg_cit_per_person',
                                                      'conf_series_citation', 'private_registration', 'identity_hidden', 'completeness',
                                                      'geo_na_eu', 'geo_asia', 'whois_na_eu', 'whois_asia', 'geo_loc_diff_whois_loc',
                                                      'website_age']), train['predatory']
test_features = test.reindex(columns=['touristic_focus', 'text_length', 'trusted_tld', 'adj_percent', 'suspicious_words_count',
                                                      'start_subDl_duration_days', 'neg_duration', 'committee_number', 'total_committee_citation',
                                                      'total_committee_publications', 'total_committee_avg_citation', 'avg_cit_per_person',
                                                      'conf_series_citation', 'private_registration', 'identity_hidden', 'completeness',
                                                      'geo_na_eu', 'geo_asia', 'whois_na_eu', 'whois_asia', 'geo_loc_diff_whois_loc',
                                                      'website_age'])
print('Usable Features of train set')
print(train_features.columns)
print(train_features.head())

cross_val = StratifiedKFold(n_splits=10, random_state=42)

# Uncomment any one classifier to use it for training
# clf = LogisticRegression()
# params = [{'solver': ['newton-cg', 'lbfgs'], 'max_iter': [100, 500, 1000], 'penalty':['l2'], 'C':[10]}]
# params = [{'solver': ['liblinear', 'saga'], 'max_iter': [500, 1000], 'penalty':['l1'], 'C':[0.001, 0.1, 1, 10]}]
#
# clf = SVC()
# params = [{'decision_function_shape': ['ovo'], 'C': [0.01, 0.5, 1], 'kernel': ['rbf', 'linear'], 'probability': [True]}]
#
# clf = RandomForestClassifier()
# params = [{'n_estimators': [1000], 'max_depth': [10], 'min_samples_split': [3], 'max_features': [4]}]
#
# clf = KNeighborsClassifier()
# params = [{'algorithm': ['auto', 'ball_tree', 'kd_tree'], 'n_neighbors': [5, 7, 10], 'metric':['euclidean', 'manhattan'], 'weights':['uniform', 'distance']}]
#
# clf = SGDClassifier()
# params = [{'epsilon': [0.1, 0.5], 'loss': ['log','modified_huber'], 'penalty': ['l1', 'l2']}]

# Running a gridsearch for a given model and parameterset
clf = GradientBoostingClassifier()
params = [{'learning_rate': [0.1], 'loss': ['exponential'], 'n_estimators': [75], 'subsample':[0.5], 'max_depth':[10]}]

print('\n Assessing for all data:')
best_clf, best_comb, best_y_pred, best_y_proba_pred = \
    mte.some_gs_Funk(clf, train_features, train_target, cross_val, params, verbose=False)

print('Feature importance for classifier - ')
try:
    for column, feature_importance in zip(train_features.columns, best_clf.feature_importances_):
        print(column, feature_importance)
except Exception as e:
    print(Exception.__name__)

dump(best_clf, '../models/classifier_GB.joblib')

# Analysing Logistic regression as base classifier to check significance of generated features
logit_model=sm.Logit(np.array(train_target, dtype=float),np.array(train_features, dtype=float))
result=logit_model.fit()
print('Logit statsmodel')
print(result.summary())

# Uncomment below code to train meta classifier
'''
# Running a grid search for various models and parametersets in order to use resulting probabilities as data for meta classifier
clf_list = [GradientBoostingClassifier(), RandomForestClassifier(), AdaBoostClassifier()]
params_list = [[{'learning_rate': [0.1], 'loss': ['exponential'], 'n_estimators':[100], 'subsample':[0.8], 'max_depth':[10]}],
               [{'n_estimators': [100], 'max_depth': [10], 'max_features': ['sqrt'], 'min_samples_split': [3]}],
               [{'learning_rate': [0.5], 'n_estimators': [500], 'random_state': [0]}]]

data_list = [train_features, train_features, train_features]
meta_data_clf = pd.DataFrame()
i = 1
print('\n Assessing level 1 models:')
for clf, params, data in zip(clf_list, params_list, data_list):
    print('\n Running classifier: %i' %i)

    best_clf, best_comb, best_y_pred, best_y_proba_pred = \
        mte.some_gs_Funk(clf, data, train_target, cross_val, params, verbose=False,
                             profit_relevant=False)

    print('Feature importance for classifier %i - ' %i)

    try:
        for column, feature_importance in zip(data.columns, best_clf.feature_importances_):
            print(column, feature_importance)
    except Exception as e:
        print(Exception.__name__)

    meta_data_clf['Classifier %i' %i] = best_y_proba_pred
    i += 1

# Building ensemble model
for column in meta_data_clf:
    mean = meta_data_clf[column].mean()
    sd = meta_data_clf[column].std()
    meta_data_clf[column] = (meta_data_clf[column] - mean) / sd

# Combine data for stacking including the original features (if to be used)
comb_data = pd.concat([meta_data_clf, train_features], axis=1, ignore_index=True)

# Choose/ initialise meta classifier and its parameters for grid search
# meta_clf = LogisticRegression()
# meta_params = [{'solver': ['newton-cg'], 'max_iter': [1000], 'multi_class': ['multinomial']}]

# meta_clf = RandomForestClassifier()
# meta_params = [{'n_estimators': [100, 1000], 'max_depth': [5, 10], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [3, 5]}]

meta_clf = GradientBoostingClassifier()
meta_params = [{'learning_rate': [0.003, 0.01, 0.1], 'loss': ['exponential', 'deviance'], 'n_estimators':[75, 100], 'subsample':[0.5, 0.8]}]

print('\n Assessing meta classifier:')
best_clf, best_comb, best_y_pred, best_y_proba_pred = mte.some_gs_Funk(meta_clf, meta_data_clf,
                                                                                  train_target, cross_val, meta_params,
                                                                                  verbose=False, profit_relevant=False)


dump(best_clf, '../models/classifier_ensemble.joblib')

print('\n------------End of Classification-----------\n')
'''
# Add predictions of Classifier model to train data
train['predatory_model1'] = best_y_pred

# Extract Non-predatory predictions
train_reg = train[train['predatory_model1'] == 0]

# Replace missing values of Field Citation Ratio as 0, as missing values imply conference not found in Dimensions
train_reg['fcr'] = train_reg['fcr'].replace(np.nan, 0)
print('Mean value of FCR {}'.format(train_reg['fcr'].mean()))
print('Std value of FCR {}'.format(train_reg['fcr'].std()))
print('Max value of FCR {}'.format(train_reg['fcr'].max()))

print(train_reg.isna().sum())
print('Training samples for Level 2:'.format(train_reg.shape[0]))

# Get features and target variable for training regression model
train_features_reg, train_target_reg = train_reg.reindex(columns=['touristic_focus', 'text_length', 'trusted_tld', 'adj_percent', 'suspicious_words_count',
                                                      'start_subDl_duration_days', 'neg_duration', 'committee_number', 'total_committee_citation',
                                                      'total_committee_publications', 'total_committee_avg_citation', 'avg_cit_per_person',
                                                      'conf_series_citation', 'private_registration', 'identity_hidden', 'completeness',
                                                      'geo_na_eu', 'geo_asia', 'whois_na_eu', 'whois_asia', 'geo_loc_diff_whois_loc',
                                                      'website_age']), train_reg['fcr']
print('Features of train set')
print(train_features_reg.columns)
print(train_features_reg.head())

# Uncomment below code to normalise train data
'''
train_features_reg['text_length'] = (train_features_reg['text_length']-train_features_reg['text_length'].mean())/train_features_reg['text_length'].std()
train_features_reg['start_subDl_duration_days'] = (train_features_reg['start_subDl_duration_days']-train_features_reg['start_subDl_duration_days'].mean())/train_features_reg['start_subDl_duration_days'].std()
train_features_reg['suspicious_words_count'] = (train_features_reg['suspicious_words_count']-train_features_reg['suspicious_words_count'].mean())/train_features_reg['suspicious_words_count'].std()
train_features_reg['total_committee_citation'] = (train_features_reg['total_committee_citation']-train_features_reg['total_committee_citation'].mean())/train_features_reg['total_committee_citation'].std()
train_features_reg['total_committee_publications'] = (train_features_reg['total_committee_publications']-train_features_reg['total_committee_publications'].mean())/train_features_reg['total_committee_publications'].std()
train_features_reg['total_committee_avg_citation'] = (train_features_reg['total_committee_avg_citation']-train_features_reg['total_committee_avg_citation'].mean())/train_features_reg['total_committee_avg_citation'].std()
train_features_reg['avg_cit_per_person'] = (train_features_reg['avg_cit_per_person']-train_features_reg['avg_cit_per_person'].mean())/train_features_reg['avg_cit_per_person'].std()
train_features_reg['conf_series_citation'] = (train_features_reg['conf_series_citation']-train_features_reg['conf_series_citation'].mean())/train_features_reg['conf_series_citation'].std()
train_features_reg['website_age'] = (train_features_reg['website_age']-train['website_age'].mean())/train_features_reg['website_age'].std()
'''
cross_val_reg = KFold(n_splits=10, random_state=42)

# Uncomment any one classifier to train regession model
# reg = LinearRegression()
# params_reg = [{'fit_intercept': [True], 'normalize': [True]}]
#
# reg = Ridge()
# params_reg = [{'alpha': [0.1, 1, 2], 'fit_intercept': [True, False], 'solver':['auto', 'svd', 'sag']}]
#
# reg = DecisionTreeRegressor()
# params_reg = [{'criterion': ['mse', 'friedman_mse'], 'max_depth': [None, 5, 7, 10], 'max_features':[None, 'auto', 'sqrt', 'log2'],
#                'max_leaf_nodes':[None, 50, 100, 150], 'min_samples_leaf':[1, 2, 3]}]
#
# reg = xgb.XGBRegressor()
# params_reg = [{'colsample_bytree': [0.3, 0.6, 0.8, 1], 'gamma': [1], 'learning_rate':[0.009, 0.01, 0.03], 'max_depth':[3],
#                'n_estimators':[50, 100, 250], 'subsample':[0.8]}]
#
# reg = KernelRidge()
# params_reg = [{'alpha': [1, 2], 'kernel': ['linear', 'sigmoid'], 'gamma':[0.1, 3.0, 1], 'degree':[3, 4 ,5]}]

reg = RandomForestRegressor()
params_reg = [{'n_estimators': [1000], 'max_depth': [10], 'max_features': ['sqrt'], 'min_samples_split': [3]}]


print('\n Assessing for all data:')
best_reg, best_comb_reg, best_y_pred_reg = \
    mte.some_gs_Funk_reg(reg, train_features_reg, train_target_reg, cross_val_reg, params_reg, verbose=False, r2_score_relevant=False)

try:
    for column, feature_importance in zip(train_features_reg.columns, best_reg.feature_importances_):
        print(column, feature_importance)
except Exception as e:
    print(Exception.__name__)

dump(best_reg, '../models/regressor_RF.joblib')

# Uncomment below code to train meta regression model

# Running a grid search for various models and parametersets in order to use resulting probabilities as data for meta classifier
# reg_list = [GradientBoostingRegressor(), xgb.XGBRegressor(), RandomForestRegressor()]
# params_reg_list = [[{'loss': ['huber'], 'learning_rate': [0.03], 'n_estimators':[1000], 'subsample':[0.8]}],
#                    [{'gamma': [1], 'learning_rate': [0.1], 'max_depth': [3], 'n_estimators':[100], 'subsample':[0.8]}],
#                    [{'n_estimators': [1000], 'max_depth': [10], 'max_features': ['sqrt'],
#                      'min_samples_split': [3]}]]
# data_reg_list = [train_features_reg, train_features_reg, train_features_reg]
# meta_data_reg = pd.DataFrame()
# i = 1
# print('\n Assessing level 1 models:')
# for reg, params_reg, data_reg in zip(reg_list, params_reg_list, data_reg_list):
#     print('\n Running regressor: %i' %i)
#
#     best_reg, best_comb_reg, best_y_pred_reg = \
#         mte.some_gs_Funk_reg(reg, data_reg, train_target_reg, cross_val_reg, params_reg, verbose=False,
#                              r2_score_relevant=False)
#
#     print('Feature importance for regressor %i - ' %i)
#
#     try:
#         for column, feature_importance in zip(data_reg.columns, best_reg.feature_importances_):
#             print(column, feature_importance)
#     except Exception as e:
#         print(Exception.__name__)
#
#     meta_data_reg['Regressor %i' %i] = best_y_pred_reg
#     i += 1
#
# for column in meta_data_reg:
#     mean = meta_data_reg[column].mean()
#     sd = meta_data_reg[column].std()
#     meta_data_reg[column] = (meta_data_reg[column] - mean) / sd
#
# # Combine data for stacking including the original features (if to be used)
# comb_data = pd.concat([meta_data_reg, train_target_reg], axis=1, ignore_index=True)
#
# # Choose/ initialise meta classifier and its parameters for grid search
# # meta_reg = xgb.XGBRegressor()
# # meta_params_reg = [{'gamma': [1], 'learning_rate': [0.03], 'max_depth':[3],
# #                     'n_estimators':[250], 'subsample':[0.8]}]
#
# meta_reg = RandomForestRegressor()
# meta_params_reg = [{'n_estimators': [100, 1000], 'max_depth': [5, 10], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [3, 5]}]
#
# print('\n Assessing meta regressor:')
# best_reg, best_comb_reg, best_y_pred_reg = mte.some_gs_Funk_reg(meta_reg, meta_data_reg, train_target_reg, cross_val_reg, meta_params_reg, verbose=False)
#
# dump(best_reg, '../models/regressor_ensemble.joblib')


print('\n------------End of Regression-----------\n')

print("Code executed in - {}".format(time.time() - start_time))