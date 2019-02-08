import numpy as np
import sys
import argparse
import os
import json
import csv
import string

# region Constants and Builder Functions

category_numbers = {
    "Left": 0,
    "Center": 1,
    "Right": 2,
    "Alt": 3
}

# TODO REFACTOR INTO SIBLING CONSTANTS FILE?
# TODO CHANGE THIS TO RUN ON CDF
BGL_PATH = "../wordlists/BristolNorms+GilhoolyLogie.csv"


def build_bgl_stats():
    """
    TODO DOCSTRING
    """
    with open(BGL_PATH, "r") as csv_file:
        return {
            row["WORD"]: row
            for row in csv.DictReader(csv_file)
        }


# TODO CHANGE THIS TO RUN ON CDF
WAR_PATH = "../wordlists/Ratings_Warriner_et_al.csv"


def build_war_stats():
    """
    TODO DOCSTRING
    """
    with open(WAR_PATH, "r") as csv_file:
        return {
            row["Word"]: row
            for row in csv.DictReader(csv_file)
        }


def build_id_to_liwc_feats():
    """
    TODO DOCSTRING
    :return:
    """

    ret = {}

    for category in ["Alt", "Center", "Left", "Right"]:

        # TODO CHANGE TO RUN ON CDF
        with open(f"../feats/{category}_IDs.txt", "r") as id_file:
            liwc_features = np.load(f"../feats/{category}_feats.dat.npy")

            for (row_num, id_line) in enumerate(id_file.readlines()):
                ret[id_line.strip()] = liwc_features[row_num]

    return ret


bgl_stats = build_bgl_stats()

war_stats = build_war_stats()

id_to_liwc_feats = build_id_to_liwc_feats()

punc_chars = set(string.punctuation)


# endregion

# region Helper Functions

def get_sentences(comment):
    """
    TODO DOCSTRING
    :param comment:
    :return:
    """
    return (
        x.strip()
        for x in comment.split("\n")
    )


def last_forward_slash_ind(tagged_word):
    """
    TODO DOCSTRING
    :param tagged_word:
    :return:
    """

    for i in range(1, len(tagged_word) + 1):
        char = tagged_word[-i]
        if char == "/":
            return len(tagged_word) - i

    return -1


def split_tagged_word(tag_word):
    """
    TODO DOCSTRING
    :param tag_word:
    :return:
    """
    slash_ind = last_forward_slash_ind(tag_word)

    if slash_ind == -1:
        word = tag_word
        tag = None
    else:
        word = tag_word[:slash_ind]
        tag = tag_word[slash_ind + 1:]

    return (word, tag)


def build_word_counter(words):
    """([str]) -> (([[(str, str?)]] -> int))
    Returns a function that counts the number of
    words whose lowercase form is in the
    lowercase set of words.

    The function takes tagged sentences
    of form [[(word, tag)]] as input, and counts the
    number of times word.lower() appears in words (lower-cased)

    :param words: iterable of word strings
    """

    assert type(words) is not str

    word_set = set(x.lower() for x in words)

    def ret_func(tagged_sents):
        return sum(
            1
            for sent in tagged_sents
            for (word, tag) in sent
            if word.lower() in word_set
        )

    return ret_func


def build_tag_counter(tags):
    """([str]) -> (([[(str, str?)]] -> int))

    Returns a function that counts the number of
    tagged words whose lower-cased tag is in the
    lowercase set of tags.

    The function takes tagged sentences
    of form [[(word, tag)]] as input, and counts the
    number of times tag.lower() appears in tags (lower-cased)

    :param tags: iterable of tags strings
    """

    assert type(tags) is not str

    tag_set = set(x.lower() for x in tags)

    def ret_func(tagged_sents):
        return sum(
            1
            for sent in tagged_sents
            for (word, tag) in sent
            if tag and tag.lower() in tag_set
        )

    return ret_func


def get_valid_word_statistics(tagged_sents, word_to_stat):
    """
    TODO DOCSTRING
    :param tagged_sents:
    :param word_to_stat:
    :return:
    """

    stats = []
    for sent in tagged_sents:
        for (word, tag) in sent:
            try:
                stats += [word_to_stat(word)]
            except:
                pass

    return stats


def build_word_avg(word_to_stat):
    """
    TODO DOCSTRING
    :param word_to_stat:
    :return:
    """

    def ret_func(tagged_sents):
        arr = get_valid_word_statistics(tagged_sents,
                                        word_to_stat)
        if arr:
            return np.mean(arr)
        else:
            return 0

    return ret_func


def build_word_std(word_to_stat):
    """
    TODO DOCSTRING
    :param word_to_stat:
    :return:
    """

    def ret_func(tagged_sents):

        arr = get_valid_word_statistics(tagged_sents,
                                        word_to_stat)

        if arr:
            return np.std(arr)
        else:
            return 0

    return ret_func


# endregion

# region Statistics Functions


# First-person pronouns given in assignment handout
num_1p_pronouns = build_word_counter([
    "I", "me", "my", "mine", "we", "us", "our", "ours"
])

# Second person pronouns given in assignment handout
num_2p_pronouns = build_word_counter([
    'you', 'your', 'yours', 'u', 'ur', 'urs'
])

# Third person pronouns given in assignment handout
num_3p_pronouns = build_word_counter([
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "they", "them", "their", "theirs"
])

# Coordinating conjunctions are tagged with CC
num_coord_conjunctions = build_tag_counter(["CC"])

# Past tense verbs are tagged with VBD
num_past_verbs = build_tag_counter(["VBD"])


# TODO REQUIRES PATTERN MATCHING
def num_future_verbs(tagged_sents):
    """
    TODO DOCSTRING
    """
    ret = 0
    for sent in tagged_sents:
        for (ind, (word, tag)) in enumerate(sent):

            # Single words given in handout
            if word.lower() in ["'ll", "will", "gonna"]:
                ret += 1

            # Multi-word form given in handout (going+to+VB),
            # need to look backwards by 2 indices
            if (tag == "VB") and ind > 1:
                start_ind, end_ind = ind - 2, ind

                # Two preceding elements before the current element
                (word1, tag1), (word2, tag2) = sent[start_ind:end_ind]

                if word1 == "going" and word2 is "to":
                    ret += 1

    return ret


# Commas are tagged with ","
num_commas = build_tag_counter([","])


def num_multi_punc(tagged_sents):
    return sum(
        1
        for sent in tagged_sents
        for (word, tag) in sent
        if ((len(word) > 1) and
            all(char in punc_chars for char in word))
    )


# Common nouns are tagged with NN, NNS
num_common_nouns = build_tag_counter(["NN", "NNS"])

# Proper nouns are tagged with NNP, NNPS
num_proper_nouns = build_tag_counter(["NNP", "NNPS"])

# Adverbs are tagged with RB, RBR, RBS
num_adverbs = build_tag_counter(["RB", "RBR", "RBS"])

# Wh words are tagged with WDT, WP, WP$, WRB
num_wh_words = build_tag_counter(["WDT", "WP", "WP$", "WRB"])

# Slang acronyms given in assignment handout
num_slang_acro = build_word_counter([
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh',
    'rofl', 'wtf', 'bff', 'wyd', 'lylc', 'brb',
    'atm', 'imao', 'sml', 'btw', 'bw', 'imho',
    'fyi', 'ppl', 'sob', 'ttyl', 'imo', 'ltr',
    'thx', 'kk', 'omg', 'omfg', 'ttys', 'afn',
    'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk',
    'k', 'ly', 'ya', 'nm', 'np', 'plz', 'ru',
    'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'
])


def num_big_caps_words(tagged_sents):
    """([[(str, str)]]) -> int

    :param tagged_sents: list of form [[(word, tag)]]

    :return the number of uppercase words
    that are at least 3 letters long in tagged_sents.
    """
    return sum(
        1
        for sent in tagged_sents
        for (word, tag) in sent
        if word.isupper() and len(word) >= 3
    )


def avg_sent_length(tagged_sents):
    """([[(str, str)]]) -> int

    :param tagged_sents: list of form [[(word, tag)]]

    :return: average sentence length
    (in terms of tokens / words) of the
    sentences in tagged_sents.
    """

    total = sum(
        len(word)
        for sent in tagged_sents
        for (word, tag) in sent
    )

    return total / len(tagged_sents)


def avg_token_length(tagged_sents):
    """
    TODO DOCSTRING

    Excludes punctuation-only tokens
    :param tagged_sents:
    :return:
    """

    tok_lengths = [
        len(word)
        for sent in tagged_sents
        for (word, tag) in sent
        if any(char not in punc_chars for char in word)
    ]

    if tok_lengths:
        return np.mean(tok_lengths)
    else:
        return 0


def num_sentences(tagged_sents):
    """([[(str, str)]]) -> int

    Return the number of sentences in tagged_sents
    (the length).

    :param tagged_sents: list of form [[(word, tag)]]
    :return: length of tagged_sents
    """
    return len(tagged_sents)


avg_bgl_aoa = build_word_avg(
    lambda word: float(bgl_stats[word.lower()]["AoA (100-700)"])
)

avg_bgl_img = build_word_avg(
    lambda word: float(bgl_stats[word.lower()]["IMG"])
)

avg_bgl_fam = build_word_avg(
    lambda word: float(bgl_stats[word.lower()]["FAM"])
)

sd_bgl_aoa = build_word_std(
    lambda word: float(bgl_stats[word.lower()]["AoA (100-700)"])
)

sd_bgl_img = build_word_std(
    lambda word: float(bgl_stats[word.lower()]["IMG"])
)

sd_bgl_fam = build_word_std(
    lambda word: float(bgl_stats[word.lower()]["FAM"])
)

avg_war_vms = build_word_avg(
    lambda word: float(war_stats[word]["V.Mean.Sum"])
)

avg_war_ams = build_word_avg(
    lambda word: float(war_stats[word]["A.Mean.Sum"])
)

avg_war_dms = build_word_avg(
    lambda word: float(war_stats[word]["D.Mean.Sum"])
)

sd_war_vms = build_word_std(
    lambda word: float(war_stats[word]["V.Mean.Sum"])
)

sd_war_ams = build_word_std(
    lambda word: float(war_stats[word]["A.Mean.Sum"])
)

sd_war_dms = build_word_std(
    lambda word: float(war_stats[word]["D.Mean.Sum"])
)

# Attributes as functions, in assignment order
attributes = [
    num_1p_pronouns,
    num_2p_pronouns,
    num_3p_pronouns,
    num_coord_conjunctions,
    num_past_verbs,
    num_future_verbs,
    num_commas,
    num_multi_punc,
    num_common_nouns,
    num_proper_nouns,
    num_adverbs,
    num_wh_words,
    num_slang_acro,
    num_big_caps_words,
    avg_sent_length,
    avg_token_length,
    num_sentences,
    avg_bgl_aoa,
    avg_bgl_img,
    avg_bgl_fam,
    sd_bgl_aoa,
    sd_bgl_img,
    sd_bgl_fam,
    avg_war_vms,
    avg_war_ams,
    avg_war_dms,
    sd_war_vms,
    sd_war_ams,
    sd_war_dms,
]


# endregion


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features
        (only the first 29 are expected to be filled, here)
    '''

    tagged_sents = [
        [
            split_tagged_word(tag_word)
            for tag_word in sent.split(" ")
        ]
        for sent in get_sentences(comment["body"])
    ]

    single_features = [
        attr(tagged_sents)
        for attr in attributes
    ]

    ret = np.zeros(173, dtype=np.float)
    ret[:29] = single_features

    # Fill in LIWC (by comment ID)
    ret[29:] = id_to_liwc_feats[comment["id"]]

    return ret


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    for (row_num, comment) in enumerate(data):

        # Get the row (LIWC features are already filled in)
        row = extract1(comment)

        # Put category on end of row, then put row into slot
        feats[row_num] = np.concatenate((
            row,
            [category_numbers[comment["cat"]]]
        ))

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')

    parser.add_argument("-o",
                        "--output",
                        help="Directs the output to a filename of your choice", required=True)

    parser.add_argument("-i",
                        "--input",
                        help="The input JSON file, preprocessed as in Task 1", required=True)

    args = parser.parse_args()

    main(args)
