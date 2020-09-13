import json
import pandas as pd


def print_publ_short(publ):
    print('\n\nNew Publications')
    print('Title: %s' % publ['title'])
    print('Date: %s' % publ['date'])
    pro_title = 'n/a'
    if 'proceedings_title' in publ:
        pro_title = publ['proceedings_title']
    print('Proceedings Title: %s' % pro_title)

    res_print = 'n/a'
    if 'researchers' in publ:
        researchers = []
        for elem in publ['researchers']:
            if 'last_name' in elem:
                last_name = elem['last_name']
            else:
                last_name = 'n/a'
            if 'first_name' in elem:
                first_name = elem['first_name']
            else:
                first_name = 'n/a'
            if last_name != 'n/a' or first_name != 'n/a':
                researchers.append(last_name + ', ' + first_name)
        res_print = '; '.join(researchers)
    elif 'author_affiliations' in publ:
        researchers = []
        for elem in publ['author_affiliations'][0]:
            if 'last_name' in elem:
                last_name = elem['last_name']
            else:
                last_name = 'n/a'
            if 'first_name' in elem:
                first_name = elem['first_name']
            else:
                first_name = 'n/a'
            if last_name != 'n/a' or first_name != 'n/a':
                researchers.append(last_name + ', ' + first_name)
        res_print = '; '.join(researchers)
    print('Researchers: %s' % res_print)
    print('-------------')

    print('Citations: %s (t/o recent: %s)' % (publ['times_cited'], publ['recent_citations']))
    fcr = 'n/a'
    if 'field_citation_ratio' in publ:
        fcr = publ['field_citation_ratio']
    print('Field citation ratio: %s' % fcr)
    print('-------------')

    cat_print = 'n/a'
    if 'FOR' in publ:
        categories = []
        for elem in publ['FOR']:
            categories.append(elem['name'])
        cat_print = '; '.join(categories)
    print('Categories: %s' % cat_print)

    conc_print = 'n/a'
    if 'concepts' in publ:
        concepts = []
        for elem in publ['concepts']:
            concepts.append(elem)
        conc_print = '; '.join(concepts)
    print('Concepts: %s' % conc_print)

