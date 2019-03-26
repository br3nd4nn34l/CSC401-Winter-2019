import re
import string


def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    # First thing to do is strip all whitespace
    steps = [lambda sent: sent.strip()]

    # French requires that we split contractions first
    if language == "f":
        steps += [split_french_contractions]

    # Rest of steps, in order
    steps += [
        lambda sent : sent.lower(),
        split_punctuation,
        mark_start_end
    ]

    # Apply each step sequentially to the
    # sentence, then return the final output
    out_sentence = in_sentence
    for func in steps:
        out_sentence = func(out_sentence)
    return out_sentence


# region Constants and Builders

# Single characters that we can split on
splittable_punc_chars = {

    # End of sentence punctuation
    ".", "!", "?",
    # Commas
    ",",
    # Colons
    ":",
    # Semicolons
    ";",
    # Parentheses
    "(", ")",

    # Dashes between parenthesis is a red herring,
    # covered by mathematical operators:
    # https://piazza.com/class/jpzxwr3vuqa1tn?cid=352
    "+", "-", "<", ">", "=",

    # Quotation marks
    '"'
}


def build_fr_contraction_splitter():
    """
    Builds a regex used to split french contractions.
    """

    # List of contractions given in assignment handout
    contractions = [

        # Singular definite contraction
        "l'",

        # E-Muet (Silent E) contractions
        "c'", "j'", "t'",

        # Que contraction
        "qu'",

        # Puisque and lorsque contraction
        "puisqu'", "lorsqu'"
    ]

    # Make OR statement where each contraction
    # comes after a word boundary
    to_compile = "|".join(
        f"\\b{con}"
        for con in contractions
    )

    # Wrap in brackets so that re.split() keeps delimiters
    to_compile = f"({to_compile})"

    # Compile the regex to ignore casing
    return re.compile(to_compile,
                      flags=re.IGNORECASE)


# For splitting French contractions
fr_contraction_splitter = build_fr_contraction_splitter()


# endregion

# region Helpers

def split_french_contractions(in_sentence):
    """
    Splits the French contractions of in_sentence,
    in the manner described in the assignment handout.
    """
    # Split on the basis of the french contraction regex,
    # then join together with spaces
    return " ".join(
        fr_contraction_splitter.split(in_sentence)
    )


def split_punctuation(in_sentence):
    """
    Splits the punctuation of in_sentence,
    in the manner described in the assignment handout
    """

    # Indices punctuation characters that we can split on
    # (ignore everything after the end of string punctuation)
    split_inds = set(
        ind
        for (ind, char) in enumerate(in_sentence)
        if (char in splittable_punc_chars)
    )

    return "".join(
        f" {char} " if (ind in split_inds)
        else char
        for (ind, char) in enumerate(in_sentence)
    )


def mark_start_end(in_sentence):
    """
    Marks in_sentence with SENTSTART and SENTEND
    """
    return f"SENTSTART {in_sentence} SENTEND"

# endregion
