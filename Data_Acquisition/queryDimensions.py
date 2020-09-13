import dimcli
from dimcli.shortcuts import dslquery_json as dslquery_new, dslqueryall

dsl = dimcli.Dsl()

# firstName = '"Kaoru"'
# lastName = '"Tominaga"'

# firstName = '"heiko"'
# lastName = '"Paulheim"'
#
firstName = '"marcus"'
lastName = '"taft"'

# firstName = '"Andrej"'
# lastName = '"Karpathy"'

# firstName = '"rafael"'
# lastName = '"salaberry"'


search_string = """ "(artificial OR computer OR computing OR systems OR information OR
                    data OR networks OR security OR computational OR network OR dataset OR framework OR
                    internet OR cloud OR database OR IoT OR software OR server OR hardware OR robot OR
                    programming OR blockchain OR CPU)" """


def get_researcher_id(firstname1, lastname1):
    """
    Function generates a query string for get researcher ID from researcher first name and last name
    Parameters
    ----------
    firstname1: str
        First name of researcher
    lastname1: str
        Last name of researcher

    Returns
    -------
    str
        Query string

    """
    query = """search researchers
                where first_name = {} and last_name = {}
                return researchers[id]
                """.format(firstname1, lastname1)
    return query


def get_publications_from_researcher_id(researcher1):
    """
    Function generates query string to get publications data from researcher ID
    Parameters
    ----------
    researcher1 : str
        Researcher ID

    Returns
    -------
    str
        Query string

    """
    query = """
    search publications
    where researchers.id = "{}"
    return publications[times_cited + title]
     """.format(researcher1)
    return query


def get_publications_from_researcher_name(researcher_name, skip):
    """
    Function generates query string to get publication information via researcher name
    Parameters
    ----------
    researcher_name : str
        Name of researcher
    skip : int
        Offset used

    Returns
    -------
    str
        Search query

    """
    query = """
         search publications 
         in authors for "\\"{}\\""
         return publications[times_cited + concepts]
         limit 1000 skip {}
    """.format(researcher_name, skip)
    return query

