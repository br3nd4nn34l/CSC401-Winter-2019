import html
import string
import re
import unicodedata

# region Constants and Builder Functions


def build_clitic_regex():
    return re.compile("(" + "|".join(
        [
            # Clitics with starting apostraphe have trailing whitespace
            f"{left_clitic}\s+"
            for left_clitic in
            ["'d", "'n", "'ve", "'re", "'ll", "'m", "'s", "'t"]
        ] + [
            # Clitics with ending apostrophe have leading whitespace
            f"\s+{right_clitic}"
            for right_clitic in
            ["t'", "y'"]
        ] + [
            # Plural possessive clitic
            "s'\s*"
        ]
    ) + ")", flags=re.IGNORECASE)


def build_abbrev_regex():
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
        'yr.'
    ]

    return re.compile("|".join(re.escape(x) for x in abbreviations))


def build_punc_regex():
    return re.compile("|".join(
        re.escape(x) for x in string.punctuation
    ))


def build_multi_punc_regex():
    base_regex = build_punc_regex()
    return re.compile(f"({base_regex.pattern}){{2,}}")


abbrev_regex = build_abbrev_regex()

punc_regex = build_punc_regex()

multi_punc_regex = build_multi_punc_regex()

clitic_regex = build_clitic_regex()


# endregion

# region Process Steps


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
    their ASCII equivalent.

    Sources:
    https://stackoverflow.com/questions/16467479/normalizing-unicode
    https://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
    https://stackoverflow.com/questions/44431730/how-to-replace-accented-characters-in-python

    :param string: some input string
    :return: string with each HTML character converted to ASCII
    """

    unicode_str = html.unescape(string)

    return unicodedata.normalize("NFD", unicode_str) \
        .encode("ascii", "ignore") \
        .decode("utf-8") \
        .encode("ascii", "ignore")


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
    """
    TODO DOCSTRING
    :param string:
    :return:
    """

    # Determine indices of punctuation symbols
    # (that are not apostrophes)
    punc_inds = set(
        ind
        for match in punc_regex.finditer(string)
            if string[match.span()[0]] != "'"
                for ind in match.span()
    )

    # Remove anything within an abbreviation from consideration
    for match in abbrev_regex.finditer(string):
        start, end = match.span()
        for i in range(start, end + 1):
            punc_inds.discard(i)

    # Remove the indices within multiple punctuations
    for match in multi_punc_regex.finditer(string):
        start, end = match.span()
        for i in range(start + 1, end):
            punc_inds.discard(i)

    # Insert spaces in front of the targeted indices
    return "".join([
        f" {char}" if i in punc_inds else char
        for (i, char) in enumerate(string)
    ])


def split_clitics(string):
    """
    TODO DOCSTRING
    :param string:
    :return:
    """

    def clitic_replacer(match):

        # Get rid of trailing spaces for easier analysis
        clitic = match.group().strip()

        # Special case: inject space for plural possessive
        if clitic == "s'":
            return "s '"
        else:
            return f" {clitic} "

    return re.sub(clitic_regex,
                  clitic_replacer,
                  string)


def pos_tags(string):
    pass


def remove_stopwords(string):
    pass


def lemmatize(string):
    pass


def split_sentences(string):
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
        split_sentences,
        to_lowercase
    ]

    # Apply each step successively
    ret = comment
    for step in steps:
        ret = step(ret)

    return ret


print(split_punctuation("I'm sorry John. I'm afraid I can't do, that."))
