import math


def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string)
    and a list of reference sentences (list of strings).
    n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on
    
    DO NOT concatenate the measurements.
    N=2 means only bigram.
    Do not average/incorporate the uni-gram scores.
    
    INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.

	
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""

    # Split the given sentences into words
    cand_words = candidate.split()
    ref_words = [ref.split() for ref in references]

    # Calculate n-gram precision
    precision = n_gram_precision(cand_words, ref_words, n)

    # Calculate brevity penalty if needed
    if brevity:
        bp = brevity_penalty(cand_words, ref_words)
    else:
        bp = 1  # No penalty

    # BLEU = BP * P
    return bp * precision


def brevity_penalty(candidate, references):
    """
    TODO DOCSTRING

    http://www.cs.toronto.edu/~frank/csc401/lectures/6_SMT_pt3.pdf
    (Slide 41)

    :param candidate:
    :param references:
    :return:
    """

    # Reference of closest length
    closest_ref = min(
        references,
        key = lambda ref: abs(len(ref) - len(candidate))
    )

    # Length of the candidate, closest reference
    c = len(candidate)
    r = len(closest_ref)

    # Compute brevity
    brevity = r / c

    # Apply brevity penalty
    if brevity < 1:
        return 1
    else:
        return math.exp(1 - brevity)


def n_gram_precision(candidate, references, n):
    """
    TODO DOCSTRING

    http://www.cs.toronto.edu/~frank/csc401/lectures/6_SMT_pt3.pdf
    (Slide 37)

    :param candidate:
    :param references:
    :param n:
    :return:
    """

    # n-grams of the candidate
    cand_grams = list(get_n_grams(candidate, n))

    # Unique n-grams of all references
    ref_grams = set(
        gram
        for ref in references
        for gram in get_n_grams(ref, n)
    )

    # Number of n-grams in candidate
    N = len(cand_grams)

    # Number of n-grams in candidate which are in (at least) one reference
    C = sum(
        1
        for c_gram in cand_grams
        if c_gram in ref_grams
    )

    return C / N


def get_n_grams(tokens, n):
    """
    TODO DOCSTRING
    """
    last_tok_ind = len(tokens) - n
    for i in range(last_tok_ind + 1):
        yield tuple(tokens[i:i + n])
