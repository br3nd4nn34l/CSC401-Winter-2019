from preprocess import *
import pickle
import os


def get_n_grams(tokens, n):
    """
    TODO DOCSTRING
    """
    last_tok_ind = len(tokens) - n
    for i in range(last_tok_ind + 1):
        yield tuple(tokens[i:i + n])

def get_dict_branch(dct, branch):
    """
    TODO DOCSTRING
    """
    cur = dct
    for key in branch:
        cur = cur[key]
    return cur

def set_dict_branch(dct, branch, value):
    *front, last = branch

    cur = dct
    for key in front:
        if key not in cur:
            cur[key] = {}
        cur = cur[key]

    cur[last] = value


def increment_dict_branch(dct, branch, amount):
    """
    TODO DOCSTRING
    """
    try:
        cur_amt = get_dict_branch(dct, branch)
    except:
        cur_amt = 0
    set_dict_branch(dct, branch, cur_amt + amount)


def add_gram_to_lang_model(model, gram):
    """
    TODO DOCSTRING
    :param model:
    :param gram:
    :return:
    """

    # Construct the branch to descend
    branch = []
    if len(gram) == 1:
        branch += ["uni"]
    elif len(gram) == 2:
        branch += ["bi"]
    branch += list(gram)

    # Increment the count by one
    increment_dict_branch(model, branch, 1)


def add_file_to_lang_model(model, file_path, language):
    """
    TODO DOCSTRING
    """
    with open(file_path, "r") as lang_file:
        for sentence in lang_file.readlines():
            tokens = preprocess(sentence, language).split()

            # Add 1-grams and 2-grams of the pre-processed sentence
            for n in [1, 2]:
                for n_gram in get_n_grams(tokens, n):
                    add_gram_to_lang_model(model, n_gram)


def get_lang_file_paths(data_dir, language):
    """
    TODO DOCSTRING
    :param data_dir:
    :param language:
    :return:
    """
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)

        # Only consider files
        if not os.path.isfile(file_path):
            continue

        # That end in language
        extension = os.path.splitext(file_path)[1][1:]  # Remove . in extension
        if extension != language:
            continue

        # Path should be valid, yield it
        yield file_path


def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """

    language_model = {}

    # Add each file path
    for file_path in get_lang_file_paths(data_dir, language):
        add_file_to_lang_model(language_model, file_path, language)

    # Save Model
    with open(fn_LM + ".pickle", 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return language_model
