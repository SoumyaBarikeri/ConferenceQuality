import csv
import pandas as pd
# import grammar_check as gc
from spellchecker import SpellChecker
import nltk
import nltk.tag.stanford as stag
import time

start_time = time.time()


df = pd.read_csv('../data/wCfP_data_full_new.csv')
df = df.head(30)
# df.set_index('eventID')

# tool = gc.LanguageTool('en-GB')

spell = SpellChecker(distance=2)
spell.word_frequency.load_words(['http', 'google', 'metadata', 'https', 'url', 'gmail','website', 'wordnet', 'ontologies'])
# spell.known(['http', 'microsoft', 'google'])

tagger = stag.StanfordNERTagger('/Users/soumya/Documents/Mannheim-Data-Science/Sem 2/Team project/WikiCfp/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz', '/Users/soumya/Documents/Mannheim-Data-Science/Sem 2/Team project/WikiCfp/stanford-ner-2018-10-16/stanford-ner.jar')


# for word in misspelled:
#     # Get the one `most likely` answer
#     print(spell.correction(word))
#
#     # Get a list of `likely` options
#     print(spell.candidates(word))

for index, row in df.iterrows():
    conf_text = row['text']
    # matches = tool.check(text)
    # print(len(matches))
    tokens = nltk.word_tokenize(conf_text)
    tokens_clean = [word for word in tokens if word.isalpha()] # remove numbers and punctuations
    tokens_tagged = tagger.tag(tokens_clean)
    tokens_not_person_org = [s[0] for s in tokens_tagged if s[1] != 'PERSON' and s[1] != 'ORGANIZATION'] # remove person and organisation names
    # print(tokens_not_person_org)
    # print(tokens_tagged)
    misspelled = spell.unknown(tokens_not_person_org)
    print(len(misspelled))
    print(misspelled)
    df.at[index, 'spell_errors'] = len(misspelled)

df.to_csv('/Users/soumya/PycharmProjects/QualityConferences/data/wCfp_data_full_new_spell.csv', index=False)

print("Code executed in - {}".format(time.time() - start_time))

# wrong identification - isbi, macroscale, cognizant, iscas, modeling, uploaded, channeling, biometric
# 90 seconds for 30 conf