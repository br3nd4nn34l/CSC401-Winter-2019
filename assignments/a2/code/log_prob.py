from preprocess import *
from lm_train import *
from math import log


def log2(num):
    """
    Computes the log base 2 of num.
    If num is 0, returns -infinity.
    """
    if (num == 0):
        return float("-inf")
    return log(num, 2)


def get_unigram_count(lang_model, word):
    """
    Attempts to get the unigram count of word from lang_model.
    Returns 0 if the count does not exist.
    """
    try:
        return get_dict_branch(lang_model, ["uni", word])
    except:
        return 0


def get_bigram_count(lang_model, word1, word2):
    """
    Attempts to get the bigram count of (word1, word2) from lang_model.
    Returns 0 if the count does not exist.
    """
    try:
        return get_dict_branch(lang_model, ["bi", word1, word2])
    except:
        return 0

def log_prob_w2_given_w1(lang_model, w1, w2, delta, vocabSize):
    """
    Computes the log probability of w2 given w1:
    log (P(w2|w1)) = log(P(w_t|w_{t-1}))
    """
    # Formula for bigram conditional probability given here (delta = 1):
    # http://www.cs.toronto.edu/~frank/csc401/lectures/2_Corpora_and_Smoothing.pdf
    # (slide marked 71)
    # Also given on assignment handout
    count_w1w2 = get_bigram_count(lang_model, w1, w2)
    count_w1 = get_unigram_count(lang_model, w1)

    numerator = count_w1w2 + delta
    denominator = count_w1 + (delta * vocabSize)

    # log(N/D) = log(N) - log(D)
    return log2(numerator) - log2(denominator)

def get_bigrams(sentence):
    """
    Yields the bigrams of sentence
    """
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
