"""
Functions to be used in the data acquisition pipeline
"""


def get_dimensions_data(start=2008, end=2020):
    """
    saves and returns a DataFrame with citation data of all proceedings (updated for a given timeframe)

    Parameters
    ----------
    start : int
        start is used to determine the first year for which dimensions data should be accessed
    end : int
        end is used to determine the layst year for which dimensions data should be accessed
    Returns
    -------
    df : pd.DataFrame
        The updated DataFrame of all proceedings found in dimensions
    """
    import dimcli
    import pandas as pd
    import numpy as np

    dsl = dimcli.Dsl()

    #initialise the result dataframe
    df = pd.DataFrame(columns=['title', 'citations', 'fcr', 'papers', 'date', 'pub_year'])
    df['fcr'] = df['fcr'].astype(np.float32)

    # iterate over all years for which data is required
    for year in range(start, end):
        print('\n\nStarting year: %i' % year)

        # to circumvent skipping limit of 50,000 iterate over various citation ranges
        for times_cited in ['and times_cited = 0', 'and times_cited > 0 and times_cited < 10', 'and times_cited >= 10']:
            # search string refers holds all desired concepts, limiting the request to CS publications
            search_string = """ "(artificial OR computer OR computing OR systems OR information OR
                                data OR networks OR security OR computational OR network OR dataset OR framework OR
                                internet OR cloud OR database OR IoT OR software OR server OR hardware OR robot OR
                                programming OR blockchain OR CPU)" """
            query = """search publications
                       in concepts for""" + search_string + """where year in [""" + str(year) + ":" + str(year) + """] 
                       and type="proceeding" """ + times_cited + """
                       return publications[proceedings_title + concepts + year + field_citation_ratio + times_cited + date]
                       limit 1000 skip """

            skip = 0
            # request data chunks through skip as long as data is returned
            while True:
                data = dsl.query(query + str(skip))
                if len(data['publications']) == 0:
                    # stop when no more publications are returned
                    break
                for elem in data['publications']:
                    if 'proceedings_title' not in elem:
                        # if element includes no proceedings title it cannot be matched
                        continue
                    proc_title, pub_year = elem['proceedings_title'], elem['year']
                    idx = proc_title + ' - ' + str(pub_year)
                    # if proceedings are already in dataframe increment numbers otherwise append proceedings
                    if idx in df.index:
                        if 'times_cited' in elem:
                            df.at[idx, 'citations'] += elem['times_cited']
                        if 'field_citation_ratio' in elem:
                            df.at[idx, 'fcr'] += elem['field_citation_ratio']
                        df.at[idx, 'papers'] += 1
                    else:
                        df = df.append(pd.Series(name=idx))
                        if 'times_cited' in elem:
                            df.at[idx, 'citations'] = elem['times_cited']
                        else:
                            df.at[idx, 'citations'] = 0
                        if 'field_citation_ratio' in elem:
                            df.at[idx, 'fcr'] = elem['field_citation_ratio']
                        else:
                            df.at[idx, 'fcr'] = 0
                        df.at[idx, 'papers'] = 1
                        df.at[idx, 'date'] = elem['date']
                        df.at[idx, 'title'] = proc_title
                        df.at[idx, 'pub_year'] = pub_year

                skip += 1000

    df['fcr'] = df['fcr'] / df['papers']

    df_old = pd.read_csv('../data/dim_proceedings_full_new.csv', index_col=0)

    # remove all updated rows from old data
    to_be_removed = []
    for idx in df_old.index:
        if idx in df.index:
            to_be_removed.append(idx)

    # add old data to updated data
    df_old = df_old.drop(to_be_removed)
    df = df.append(df_old)

    df.to_csv('../data/dim_proceedings_full_new.csv', index=True)

    return df
