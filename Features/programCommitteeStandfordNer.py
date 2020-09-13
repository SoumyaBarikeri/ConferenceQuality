"""
This file contains functions required for extracting Person names
"""
import nltk
import nltk.tag.stanford as stag
from nameparser.parser import HumanName
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.tree import Tree

# Please install NLTK and download corresponding files
tagger = stag.StanfordNERTagger('/Users/soumya/Documents/Mannheim-Data-Science/Sem 2/Team project/WikiCfp/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz', '/Users/soumya/Documents/Mannheim-Data-Science/Sem 2/Team project/WikiCfp/stanford-ner-2018-10-16/stanford-ner.jar')


def stanfordNE2BIO(tagged_sent):
    """
    Function converts the Named Entity tagged sentence to BIO(Beginning Inside Outside) tagged sentence

    Parameters
    ----------
    tagged_sent : list
        Sentence tagged by Standford NER tagger

    Returns
    -------
    list
        Sentence tagged in BIO format

    """
    bio_tagged_sent = []
    prev_tag = "O"
    for token, tag in tagged_sent:
        if tag == "O": #O
            bio_tagged_sent.append((token, tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O": # Begin NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag: # Inside NE
            bio_tagged_sent.append((token, "I-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag: # Adjacent NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag

    return bio_tagged_sent


def stanfordNE2tree(ne_tagged_sent):
    """
    Function converts the Named Entity tagged sentence to a tree
    Parameters
    ----------
    ne_tagged_sent : list
        Named entity tagged sentence by Standford NER tagger

    Returns
    -------
    Tree
        NLTK tree structure of CoNLL IOB

    """
    bio_tagged_sent = stanfordNE2BIO(ne_tagged_sent)
    sent_tokens, sent_ne_tags = zip(*bio_tagged_sent)
    sent_pos_tags = [pos for token, pos in pos_tag(sent_tokens)]

    sent_conlltags = [(token, pos, ne) for token, pos, ne in zip(sent_tokens, sent_pos_tags, sent_ne_tags)]
    ne_tree = conlltags2tree(sent_conlltags)
    return ne_tree


def getProgramCommittee(text):
    """
    Function extracts names of people from conference text

    Parameters
    ----------
    text : str
        Call For Papers text from WikiCfp

    Returns
    -------
    list
        Names of people as list

    """
    committeenames = []

    tokens = nltk.tokenize.word_tokenize(text)
    ne_tagged_sent = tagger.tag(tokens)
    print(ne_tagged_sent)

    ne_tree = stanfordNE2tree(ne_tagged_sent)

    ne_in_sent = []
    for subtree in ne_tree:
        if type(subtree) == Tree:
            ne_label = subtree.label()
            ne_string = " ".join([token for token, pos in subtree.leaves()])
            ne_in_sent.append((ne_string, ne_label))

    for ne in ne_in_sent:
        if ne[1] == 'PERSON':
            committeenames.append(ne[0])

    return committeenames


def getNames(text):
    """
    Function extracts names of people from conference text

    Parameters
    ----------
    text : str
        Call For Papers text from WikiCfp

    Returns
    -------
    list
        Names of people as list

    """
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)
    person_list = []
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        for part in person:
            name += part + ' '
        if name[:-1] not in person_list:
            person_list.append(name[:-1])
            name = ''
        person = []

    return person_list
