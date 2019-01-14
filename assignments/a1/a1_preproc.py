# region Imports

import html
import string

# endregion

# region Constants

clitics = [
    "'d", "'n", "'ve",
    "'re", "'ll", "'m",
    "'re", "'s",
    "t'", "y'"
]

abbreviations = [
    'Ala.',
    'Ariz.',
    'Assn.',
    'Atty.',
    'Aug.',
    'Ave.',
    'Bldg.',
    'Blvd.',
    'Calif.',
    'Capt.',
    'Cf.',
    'Ch.',
    'Co.',
    'Col.',
    'Colo.',
    'Conn.',
    'Corp.',
    'DR.',
    'Dec.',
    'Dept.',
    'Dist.',
    'Dr.',
    'Drs.',
    'Ed.',
    'Eq.',
    'FEB.',
    'Feb.',
    'Fig.',
    'Figs.',
    'Fla.',
    'Ga.',
    'Gen.',
    'Gov.',
    'HON.',
    'Ill.',
    'Inc.',
    'JR.',
    'Jan.',
    'Jr.',
    'Kan.',
    'Ky.',
    'La.',
    'Lt.',
    'Ltd.',
    'MR.',
    'MRS.',
    'Mar.',
    'Mass.',
    'Md.',
    'Messrs.',
    'Mich.',
    'Minn.',
    'Miss.',
    'Mmes.',
    'Mo.',
    'Mr.',
    'Mrs.',
    'Ms.',
    'Mx.',
    'Mt.',
    'NO.',
    'No.',
    'Nov.',
    'Oct.',
    'Okla.',
    'Op.',
    'Ore.',
    'Pa.',
    'Pp.',
    'Prof.',
    'Prop.',
    'Rd.',
    'Ref.',
    'Rep.',
    'Reps.',
    'Rev.',
    'Rte.',
    'Sen.',
    'Sept.',
    'Sr.',
    'St.',
    'Stat.',
    'Supt.',
    'Tech.',
    'Tex.',
    'Va.',
    'Vol.',
    'Wash.',
    'al.',
    'av.',
    'ave.',
    'ca.',
    'cc.',
    'chap.',
    'cm.',
    'cu.',
    'dia.',
    'dr.',
    'eqn.',
    'etc.',
    'fig.',
    'figs.',
    'ft.',
    'gm.',
    'hr.',
    'in.',
    'kc.',
    'lb.',
    'lbs.',
    'mg.',
    'ml.',
    'mm.',
    'mv.',
    'nw.',
    'oz.',
    'pl.',
    'pp.',
    'sec.',
    'sq.',
    'st.',
    'vs.',
    'yr.']

punctuation = set(string.punctuation)


# endregion

# region Helper Functions

def remove_newlines(string):
    """(str) -> str
    Replaces newlines "\n" in string with a space " "
    :param string: some input string
    :return: string where "\n" is replaced with " "
    """
    return string.replace("\n", " ")


def replace_html_codes(string):
    """(str) -> str

    Replaces all HTML character codes in string with
    their ASCII equivalent (or expanded encoding,
    if no such ASCII character exists)

    TODO SEE IF PIAZZA GETS RESOLVED

    :param string: some input string
    :return: string with each HTML character converted to ASCII
    """
    unicode_str = html.unescape(string)


def remove_urls(string):
    """
    Removes all URLs from string.

    URLs are defined as tokens (space separated)
    that start with "www" or "http".

    :param string: some input string
    :return: string, with each URL removed
    """

    # Helper predicate for clarity
    def is_url(token):
        return token[:3] == "www" or \
               token[:4] == "http"

    # Split on spaces, then join using spaces
    return " ".join([
        token
        for token in string.split(" ")
        if not is_url(token)
    ])


def split_punctuation(string):

    # Find all punctuation indices
    punc_inds = {
        i for (i, char) in enumerate(string)
        if char in punctuation
    }

    # Remove all apostrophes from consideration
    for (i, char) in enumerate(string):
        if char == "'":
            punc_inds.discard(i)

    # Remove anything within an abbreviation from consideration

    # Remove anything within multiple punctuation from consideration

    # TODO HANDLE HYPHENS

    # Put spaces between all remaining indices


    pass


def split_clitics(string):
    """
    ()[wouldn't] -> [wouldn 't]
    :param string:
    :return:
    """
    pass


def pos_tags(string):
    pass


def remove_stopwords(string):
    pass


def lemmatize(string):
    pass


def newlined_sentences(string):
    pass


def to_lowercase(string):
    return string.lower()


# endregion

def process_comment(comment):
    # Steps ordered according to handout
    steps = [
        remove_newlines,
        replace_html_codes,
        remove_urls,
        split_punctuation,
        split_clitics,
        pos_tags,
        remove_stopwords,
        newlined_sentences,
        to_lowercase
    ]

    # Apply each step successively
    ret = comment
    for step in steps:
        ret = step(ret)

    return ret
