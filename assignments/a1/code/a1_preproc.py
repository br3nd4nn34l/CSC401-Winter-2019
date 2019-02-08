import sys
import argparse
import os
import json
import html
import string
import re
import unicodedata
import spacy
from spacy.tokens import Doc

# region Constants and Builder Functions

# TODO CHANGE THIS FOR CDF
# indir = '/u/cs401/A1/data/'
indir = '../data'

# TODO CHANGE THIS FOR CDF
# wordlists_dir = "/u/cs401/Wordlists"
wordlists_dir = "../wordlists"


def build_clitic_regex():
    """
    TODO DOCSTRING
    :return:
    """
    return re.compile("(" + "|".join(
        [
            # Clitics with starting apostrophe have trailing whitespace
            f"{left_clitic}\s+"
            for left_clitic in
            ["'d", "'n", "'ve", "'re", "'ll", "'m", "'s"]
        ] + [
            # Clitics with ending apostrophe have leading whitespace
            f"\s+{right_clitic}"
            for right_clitic in
            ["t'", "y'"]
        ] + [
            "s'\s*",  # Plural possessive clitic
            "n't\s*"  # not -> n't contraction
        ]
    ) + ")", flags=re.IGNORECASE)


def build_abbrev_regex():
    """
    TODO DOCSTRING
    :return:
    """
    abbrev_path = os.path.join(wordlists_dir, "abbrev.english")
    with open(abbrev_path, "r") as abbrev_file:
        abbreviations = [
            line.strip()
            for line in abbrev_file.readlines()
        ]
        return re.compile("|".join(re.escape(x) for x in abbreviations))


def build_punc_regex():
    """
    TODO DOCSTRING
    :return:
    """
    return re.compile("|".join(
        re.escape(x) for x in string.punctuation
    ))


def build_multi_punc_regex():
    """
    TODO DOCSTRING
    :return:
    """
    base_regex = build_punc_regex()
    return re.compile(f"({base_regex.pattern}){{2,}}")


def build_nlp():
    """
    TODO DOCSTRING
    :return:
    """
    return spacy.load("en", disable=["parser", "ner"])


def build_stopwords():
    """
    TODO DOCSTRING
    :return:
    """
    stopwords_path = os.path.join(wordlists_dir, "StopWords")
    with open(stopwords_path, mode="r") as file:
        return {
            line.strip()
            for line in file.readlines()
        }


# Obtained from http://www.noah.org/wiki/RegEx_Python#URL_regex_pattern
url_regex = re.compile(
    "http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)

abbrev_regex = build_abbrev_regex()

punc_regex = build_punc_regex()

multi_punc_regex = build_multi_punc_regex()

clitic_regex = build_clitic_regex()

nlp = build_nlp()

stopwords = build_stopwords()


# endregion


# region Helper Functions

def spacy_tokens(string):
    """
    TODO DOCSTRING
    :param string:
    :return:
    """
    # Tag tokens as instructed in tutorial
    # http://www.cs.toronto.edu/~frank/csc401/tutorials/csc401_A1_tut2.pdf
    # SLIDE 16
    doc = spacy.tokens.Doc(nlp.vocab, words=string.split())
    return nlp.tagger(doc)


# endregion

# region Process Steps


def remove_newlines(string):
    """(str) -> str
    Replaces newlines "\n" in string with a space " ",
    then strips off leading and trailing whitespace.

    :param string: some input string
    :return: string where "\n" is replaced with " "

    """
    return string \
        .replace("\n", " ") \
        .strip()


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
        .encode("ascii", "ignore") \
        .decode()


def remove_urls(string):
    """
    Removes all URLs from string.

    URLs are defined as tokens (space separated)
    that start with "www" or "http".

    :param string: some input string
    :return: string, with each URL removed
    """
    return url_regex.sub(" ", string)


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


def spacy_tagging(string,
                  add_tags,
                  remove_stopwords,
                  lemmatize,
                  split_sentences,
                  to_lowercase):
    """
    TODO DOCSTRING
    :param string:
    :param add_tags:
    :param remove_stopwords:
    :param lemmatize:
    :param split_sentences:
    :param to_lowercase:
    :return:
    """

    sentences = []
    cur_sentence = []

    for token in spacy_tokens(string):

        # Skip over stopwords if asked (step 7)
        if remove_stopwords and \
                token.text.lower() in stopwords:
            continue

        # Determine the word we want (step 8)
        if lemmatize and (token.lemma_[0] != '-'):

            # Handout Pg. 4 says for lemmatization,
            # only take tokens with non-dash beginnings
            word = token.lemma_

        else:
            word = token.text

        # Determine if word should be lowercased (step 9)
        # (tags should not be lowercased!)
        # https://piazza.com/class/jpzxwr3vuqa1tn?cid=236
        if to_lowercase:
            word = word.lower()

        # Determine if we want tags (step 6)
        if add_tags:
            to_add = f"{word}/{token.tag_}"
        else:
            to_add = word

        # Add the result to the current sentence
        cur_sentence += [to_add]

        # If the token we just added was a sentence-closer (.)
        # https://spacy.io/api/annotation -> "." is a sentence closer
        if split_sentences and (token.tag_ == "."):
            sentences += [cur_sentence]
            cur_sentence = []

    # Add any remaining sentences
    sentences += [cur_sentence]

    return "\n".join(
        " ".join(sent)
        for sent in sentences
    )


# endregion

def preproc1(comment, steps=range(1, 11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    # For faster containment checking
    step_set = set(steps)

    # No steps selected means modComm is empty
    if len(step_set) == 0:
        return ""

    # Steps were selected, apply them in ORDER
    modComm = comment
    if 1 in step_set:
        modComm = remove_newlines(modComm)
    if 2 in step_set:
        modComm = replace_html_codes(modComm)
    if 3 in step_set:
        modComm = remove_urls(modComm)
    if 4 in step_set:
        modComm = split_punctuation(modComm)
    if 5 in step_set:
        modComm = split_clitics(modComm)

    # Steps 6,7,8,9 are accomplished by one function
    # (to only call spacy once)
    if any((i in step_set)
           for i in [6, 7, 8, 9, 10]):
        modComm = spacy_tagging(modComm,
                                add_tags=(6 in step_set),
                                remove_stopwords=(7 in step_set),
                                lemmatize=(8 in step_set),
                                split_sentences=(9 in step_set),
                                to_lowercase=(10 in step_set))

    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # Select appropriate args.max lines
            max_lines = args.max
            start_ind = args.ID[0] % len(data)
            end_ind = start_ind + max_lines

            for raw_line_num in range(start_ind, end_ind):
                line_num = raw_line_num % len(data)

                # Read comment given line number
                comment = json.loads(data[line_num])

                # Add category field (determined by filename)
                comment['cat'] = file

                # Replace body with processed version (use default steps)
                comment['body'] = preproc1(comment['body'])

                # Add modified comment to "allOutput"
                allOutput += [comment]

    # Refactored from original to use with statement (auto closes)
    with open(args.output, 'w') as fout:
        fout.write(json.dumps(allOutput))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')

    parser.add_argument("-o", "--output",
                        help="Directs the output to a filename of your choice",
                        type=str,
                        required=True)

    parser.add_argument("--max",
                        help="The maximum number of comments to read from each file",
                        type=int,
                        default=10000)

    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
