#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import math
import _pickle as pickle

import decode
from align_ibm1 import *
from BLEU_score import *
from lm_train import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'

discussion = """
Discussion :

There is an increase in the 1-gram BLEU-score from 
0.48 at 1K training samples to 
0.50 at 15K training samples. 
After this point the 1-gram BLEU-score decreases 
(0.49 at 30K training samples).

It should be noted that the 1-gram BLEU-score 
is an indicator of how often the model correctly 
guesses the individual words contained in the reference 
sentences. There is no penalty to having words out of order, 
making the 1-gram BLEU score a poor indicator of 
semantic accuracy (the most useful aspect of a translator). 
For example, the following would qualify as a perfect 1-gram BLEU score:

Reference: "The bus stops at the train terminal"
Candidate 1: "The train stops at the bus terminal" (complete semantic inversion)
Candidate 2: "Terminal bus train stops the at the" (gibberish) 

The increase in 1-gram BLEU-scores indicates that there is some benefit 
to training the model on more sentences. However, the model 
degrades in performance after a certain amount of training data.
Perhaps the model is over-fitting the training data - unlikely 
since all alignments were constructed using the 
same number of EM iterations, 50. It's more likely that inherent flaws in 
the model are causing this phenomenon.

BLEU-scores using longer n-grams are a better measure of 
semantic accuracy, as longer n-grams can contain more meaning.
The model's 2-Gram and 3-Gram BLEU-scores seem to oscillate 
between 0.292-0.308 and 0.125-0.143, respectively. 
These deviations are relatively minor (both ranges are about ~0.02 long), 
so this could just be statistical noise. However, it should be noted that if the 
model was an accurate representation of the translation problem, more training data 
should lead to better performance. Therefore, the model may be inherently flawed.
"""


##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model 
    """
    pickle_path = f"{fn_LM}.pickle"
    if use_cached and os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as pickle_ptr:
            return pickle.load(pickle_ptr)

    else:
        return lm_train(data_dir, language, fn_LM)


def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory containing the data
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    pickle_path = f"{fn_AM}.pickle"
    if use_cached and os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as pickle_ptr:
            return pickle.load(pickle_ptr)

    else:
        return align_ibm1(
            train_dir=data_dir,
            num_sentences=num_sent,
            max_iter=max_iter,
            fn_AM=fn_AM
        )


def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """
    # Need to actually calculate BLEU-score:
    # https://piazza.com/class/jpzxwr3vuqa1tn?cid=503
    pi_columns = [
        [
            BLEU_score(e_dec, [e_hans, e_goog], i, brevity=False)
            for (e_dec, e_hans, e_goog) in zip(eng_decoded, eng, google_refs)
        ]
        for i in range(1, n + 1)
    ]

    p_geo_means = [
        geometric_mean(p_scores)
        for p_scores in zip(*pi_columns)
    ]

    # Have to calculate brevity separately
    brev_penalties = [
        brevity_penalty(
            candidate=eng_dec.split(),
            references=[
                eng_ref.split(),
                goog_ref.split()
            ]
        )
        for (eng_dec, eng_ref, goog_ref) in zip(eng_decoded, eng, google_refs)
    ]

    return [
        brev_pen * p_mean
        for (brev_pen, p_mean) in zip(brev_penalties, p_geo_means)
    ]


def geometric_mean(values):
    """
    Computes the geometric mean of values using
    the sum-of-logs technique
    to avoid overflow/underflow issues
    """

    def ln(n):
        if n == 0:
            return float("-inf")
        return math.log(n)

    ln_lst = [ln(v) for v in values]
    avg_ln = sum(ln_lst) / len(ln_lst)
    return math.exp(avg_ln)


def load_alignment_models(train_dir, out_dir):
    """
    Attempts to load the appropriate alignment models from out_dir.
    Models are loaded into a dictionary of form {name : model}

    If a model does not exist,
    trains an alignment model using the data in train_dir,
    and pickles the model in out_dir.
    """
    ret = {}

    for num_sents in [1000, 10000, 15000, 30000]:
        alignment_name = f"Task5_{num_sents}_Sentence_Alignment"

        ret[alignment_name] = _getAM(
            train_dir, num_sents,
            fn_AM=os.path.join(out_dir, alignment_name),
            max_iter=50
        )

    return ret


def load_test_sentences(test_dir):
    """
    Loads the test sentences from test_dir, in the form
    [[(french, english_ref, google_english_ref)]]
    """
    ret = []

    for (file_name, lang) in [("Task5.f", "f"), ("Task5.e", "e"), ("Task5.google.e", "e")]:
        file_path = os.path.join(test_dir, file_name)

        with open(file_path, "r") as file_ptr:
            ret += [[
                preprocess(sent, lang)
                for sent in file_ptr.readlines()
            ]]

    return ret


def main(data_dir, test_dir, out_dir):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """
    with open("Task5.txt", "w") as task5:

        task5.write(discussion)
        task5.write("\n\n")
        task5.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

        lang_model = _getLM(
            data_dir, "e",
            fn_LM=os.path.join(out_dir, "Task5_e_Lang_Model")
        )

        task5.write("Language Model: No Smoothing\n")

        alignment_models = load_alignment_models(
            data_dir, out_dir
        )

        french_sents, \
        eng_sents, \
        goog_sents = load_test_sentences(test_dir)

        for (name, alignment) in alignment_models.items():
            task5.write(f"\n### Start of {name} Evaluation ### \n")

            decoded_sents = [
                decode.decode(f_sent, LM=lang_model, AM=alignment)
                for f_sent in french_sents
            ]

            for n in [1, 2, 3]:
                score_lst = _get_BLEU_scores(decoded_sents, eng_sents, goog_sents, n)
                avg_score = sum(score_lst) / len(score_lst)

                task5.write(f"BLEU scores with {n}-gram: (Average  {avg_score:1.4f})\n")
                for score in score_lst:
                    task5.write(f"\t{score:1.4f}")
                    task5.write("\n")

            task5.write(f"---- ### End of {name} Evaluation ### ----\n")
