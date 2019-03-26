from lm_train import *
from log_prob import *
from preprocess import *
from math import log

from collections import Counter

import os


def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""

    # Read training data
    sent_pairs = list(read_hansard(train_dir, num_sentences))

    # Initialize AM uniformly
    AM = initialize_original({}, sent_pairs)

    # Iterate between E and M steps
    for i in range(max_iter):
        em_step_original(AM, sent_pairs)

    # Save Model
    with open(fn_AM + ".pickle", 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM


# region Support Functions (EXPOSED TO TESTS)

def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider


	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

	Make sure to read the files in an aligned manner.
	"""
    return limited(
        get_paired_sentences(train_dir),
        num_sentences
    )


def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    return initialize_original({}, zip(eng, fre))


def em_step(t, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
    return em_step_original(t, zip(eng, fre))


# endregion


# region HELPER FUNCTIONS


def initialize_original(model, paired_sents):
    """
	(ORIGINAL VERSION OF initialize HANDED IN, HAD TO BE MODIFIED FOR REMARK)
	"""

    # Dictionary of form {
    #   e_word : {
    #       f_word : number of times an
    #           e_sentence containing e_word
    #           was paired to a
    #           f_sentence containing f_word
    #   }
    # }
    totals = {}

    for (e_sent, f_sent) in paired_sents:

        # Ignore SENTSTART, SENTEND
        # https://piazza.com/class/jpzxwr3vuqa1tn?cid=457
        e_word_lst = list(e_sent.split())[1:-1]
        f_word_lst = list(f_sent.split())[1:-1]

        for e_word in e_word_lst:
            for f_word in f_word_lst:
                increment_dict_branch(totals, (e_word, f_word), 1)

    for (e_word, f_dct) in totals.items():
        num_f_words = len(f_dct)
        for f_word in f_dct:
            set_dict_branch(model, (e_word, f_word), 1 / num_f_words)

    set_dict_branch(model, ("SENTSTART", "SENTSTART"), 1)
    set_dict_branch(model, ("SENTEND", "SENTEND"), 1)

    return model


def em_step_original(model, paired_sents):
    """
    (ORIGINAL VERSION OF em_step HANDED IN, HAD TO BE MODIFIED FOR REMARK)
	"""

    total = {}

    # Flipping this from pseudocode
    # to be indexed (e, f) instead of (f, e)
    t_count = {}

    for (e_sent, f_sent) in paired_sents:

        # Ignore SENTSTART, SENTEND
        # https://piazza.com/class/jpzxwr3vuqa1tn?cid=490
        e_counter = Counter(e_sent.split()[1:-1])
        f_counter = Counter(f_sent.split()[1:-1])

        for f_word in f_counter:

            denom_c = sum(
                alignment_get(model, e_word, f_word) * f_counter[f_word]
                for e_word in e_counter
            )

            for e_word in e_counter:
                P_f_given_e = alignment_get(model, e_word, f_word)

                incrementation = (P_f_given_e *
                                  f_counter[f_word] *
                                  e_counter[e_word]) / denom_c

                # Must index backwards (e, f) instead of (f, e)
                # (see note above)
                increment_dict_branch(
                    t_count, (e_word, f_word),
                    incrementation
                )

                increment_dict_branch(
                    total, (e_word,),
                    incrementation
                )

    for e_word in total:
        for f_word in t_count[e_word]:
            alignment_set(
                model,
                e_word, f_word,
                t_count[e_word][f_word] / total[e_word]
            )


def get_paired_doc_paths(data_dir):
    """
    Yields the name-aligned document paths in data_dir,
    in the form: (english, french)
    [a.e, b.e, a.f, b.f] -> (a.e, a.f), (b.e, b.f)
    """

    # All of these are of form *.e
    english_paths = get_lang_file_paths(data_dir, "e")

    # Remove the .e from the above
    prefixes = set(
        os.path.splitext(p)[0]
        for p in english_paths
    )

    # Look for the corresponding french .f file.
    # If it exists, return the pair
    for prefix in prefixes:

        e_path = f"{prefix}.e"
        f_path = f"{prefix}.f"

        if os.path.isfile(e_path) and os.path.isfile(f_path):
            yield (e_path, f_path)


def get_paired_sentences(data_dir):
    """
    Yields the aligned sentences of the documents
    in data_dir in the following form:
    (english, french)
    """
    for (e_path, f_path) in get_paired_doc_paths(data_dir):
        with open(e_path, "r") as e_file, \
                open(f_path, "r") as f_file:

            for (e_sent, f_sent) in zip(e_file.readlines(),
                                        f_file.readlines()):
                e_proc = preprocess(e_sent, "e")
                f_proc = preprocess(f_sent, "f")

                yield (e_proc, f_proc)


def limited(iterable, max_iters):
    """
    Limits an iterable to only produce max_iters items.
    """
    # Zip will continue until shortest iterable is exhausted
    for (i, x) in zip(range(max_iters), iterable):
        yield x


def alignment_get(model, e_word, f_word):
    """
    Attempts to get the value of the
    alignment for model[e_word][f_word].

    Returns 0 if no such alignment exists.
    """
    try:
        return get_dict_branch(model, (e_word, f_word))
    except:
        return 0


def alignment_set(model, e_word, f_word, value):
    """
    Sets the value of the alignment for model[e_word][f_word].
    """
    set_dict_branch(model, (e_word, f_word), value)

# endregion
