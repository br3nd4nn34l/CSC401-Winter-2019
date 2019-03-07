from preprocess import *
from lm_train import *
from math import log


def log2(num):
    """
    TODO DOCSTRING
    :param num:
    :return:
    """
    if (num == 0):
        return float("-inf")
    return log(num, 2)


def get_unigram_count(lang_model, word):
    """
    TODO DOCSTRING
    :param word:
    :param lang_model:
    :return:
    """
    try:
        return get_dict_branch(lang_model, ["uni", word])
    except:
        return 0


def get_bigram_count(lang_model, word1, word2):
    """
    TODO DOCSTRING
    :param word:
    :param lang_model:
    :return:
    """
    try:
        return get_dict_branch(lang_model, ["bi", word1, word2])
    except:
        return 0

def log_prob_w2_given_w1(lang_model, w1, w2, delta, vocabSize):
    """
    TODO DOCSTRING
    :param w1:
    :param w2:
    :param lang_model:
    :param delta:
    :param vocabSize:
    :return:
    """

    # Formula for bigram conditional probability given here (delta = 1):
    # http://www.cs.toronto.edu/~frank/csc401/lectures/2_Corpora_and_Smoothing.pdf
    # (slide marked 71)
    count_w1w2 = get_bigram_count(lang_model, w1, w2)
    count_w1 = get_unigram_count(lang_model, w1)

    numerator = count_w1w2 + delta
    denominator = count_w1 + (delta * vocabSize)

    # log(N/D) = log(N) - log(D)
    return log2(numerator) - log2(denominator)

def get_bigrams(sentence):
    tokens = list(sentence.split())
    for i in range(len(tokens) - 1):
        yield tokens[i:i+2]


def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
    if smoothing:

        # Check to see if other variables are properly defined
        assert (0 < delta <= 1)
        assert (vocabSize > 0)

        log_probs = (
            log_prob_w2_given_w1(LM, w1, w2, delta, vocabSize)
            for (w1, w2) in get_bigrams(sentence)
        )

    else:
        # No smoothing = delta is 0
        log_probs = (
            log_prob_w2_given_w1(LM, w1, w2, 0, vocabSize)
            for (w1, w2) in get_bigrams(sentence)
        )

    # Let sentence = w1...wN
    # Then P(w1...wN) = Product(P(w_i given w_i-1))
    # Then logP(w1...wN) = Sum(logP(w_i given w_i-1))
    return sum(log_probs)
