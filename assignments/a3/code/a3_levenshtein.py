import os
import numpy as np
import string
import sys

# TODO CHANGE TO RUN ON CDF
dataDir = '/u/cs401/A3/data/'
dataDir = "../data/"


def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int)
    WER,
    number of substitutions,
    insertions, and
    deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    # Note: all metrics judge in terms of r->h transformation
    # Deletions:  deletions to make r->h
    # Insertions: insertions to make r->h
    # Substitutions: substitutions to make r->h

    # To store DP computations
    cache = {}

    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            # Reference list is empty, must insert to make into target
            if i == 0:
                num_del = 0
                num_ins = j
                num_sub = 0

            # Target is empty, must delete everything in ref
            elif j == 0:
                num_del = i
                num_ins = 0
                num_sub = 0

            else:

                # r_head = r[:-1] (Deletion)
                del_r_head, ins_r_head, sub_r_head = cache[(i - 1, j)]
                r_head_dist = sum([del_r_head, ins_r_head, sub_r_head])  # Lev[i-1][j]

                # h_head = h[:-1] (Insertion)
                del_h_head, ins_h_head, sub_h_head = cache[(i, j - 1)]
                h_head_dist = sum([del_h_head, ins_h_head, sub_h_head])  # Lev[i][j-1]

                # r_head and h_head
                del_hr_head, ins_hr_head, sub_hr_head = cache[(i - 1, j - 1)]
                hr_head_dist = sum([del_hr_head, ins_hr_head, sub_hr_head])  # Lev[i-1][j-1]
                was_sub = int(r[i - 1] != h[j - 1])

                # Compare distances, find the distance source
                lev_dist = min(
                    r_head_dist + 1,  # Deletion: Lev[i-1][j] + 1
                    h_head_dist + 1,  # Insertion: Lev[i][j-1] + 1
                    hr_head_dist + was_sub  # Substitution: Lev[i-1][j-1] + (r[i] != h[i])
                )  # Lev[i][j]

                # Experienced a deletion
                if lev_dist == r_head_dist + 1:
                    num_del = del_r_head + 1
                    num_ins = ins_r_head
                    num_sub = sub_r_head

                # Experienced an insertion
                elif lev_dist == h_head_dist + 1:
                    num_del = del_h_head
                    num_ins = ins_h_head + 1
                    num_sub = sub_h_head

                else:
                    # Carry forward deletion and insertion from r[i-1], h[j-1]
                    num_del = del_hr_head
                    num_ins = ins_hr_head

                    # Track substitution
                    num_sub = (sub_hr_head) + was_sub

            # Set the cache
            cache[(i, j)] = (num_del, num_ins, num_sub)

    n_del, n_ins, n_sub = cache[(len(r), len(h))]

    if len(r) == 0:
        wer = float('inf')
    else:
        wer = (n_sub + n_ins + n_del) / len(r)

    return (wer, n_sub, n_ins, n_del)


# region Main Functionality

def get_reference_path(data_dir, speaker):
    return os.path.join(data_dir, speaker, "transcripts.txt")


def get_google_path(data_dir, speaker):
    return os.path.join(data_dir, speaker, "transcripts.Google.txt")


def get_kaldi_path(data_dir, speaker):
    return os.path.join(data_dir, speaker, "transcripts.Kaldi.txt")


def get_speaker_names(data_dir):
    """
    Yields the names of the speakers with valid transcript paths in data_dir
    (sub-directory names)
    """
    for name in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, name)):

            paths = [
                get_reference_path(data_dir, name),
                get_google_path(data_dir, name),
                get_kaldi_path(data_dir, name)
            ]

            if all(os.path.isfile(x) for x in paths):
                yield name


punc_to_remove = set(string.punctuation) - set("[]")

def preprocess_line(line):
    """
    Removes all punctuation (excluding []) from line,
    and sets all remaining letters to lowercase.

    Returns the whitespace-split version of line.
    """

    return "".join([
        char.lower() if char not in punc_to_remove else " "
        for char in line
    ]).split()


def evaluate_wer_performance(data_dir, speaker):
    """
    Evaluates the performance of the Kaldi and Google
    models on the given Speaker's transcript

    :param data_dir: path to data directory
    :param speaker: speaker's name

    :return: tuple of form (
        k_wer_lst : WER scores for the Kaldi model
        g_wer_lst : WER scores for the Google model
    )
    """

    ref_path = get_reference_path(data_dir, speaker)
    kald_path = get_kaldi_path(data_dir, speaker)
    goog_path = get_google_path(data_dir, speaker)

    with open(ref_path, mode="r") as ref_file, \
            open(kald_path, mode="r") as kald_file, \
            open(goog_path, mode="r") as goog_file:
        k_wer_lst, g_wer_lst = [], []

        for (i, (ref_line, kald_line, goog_line)) in enumerate(zip(
                ref_file, kald_file, goog_file
        )):
            ref_words = preprocess_line(ref_line)
            kald_words = preprocess_line(kald_line)
            goog_words = preprocess_line(goog_line)

            k_wer, k_sub, k_ins, k_del = Levenshtein(ref_words, kald_words)
            k_wer_lst += [k_wer]
            print(f"[{speaker}] Kaldi  {i} {k_wer:.3f} S:{k_sub}, I:{k_ins}, D:{k_del}")

            g_wer, g_sub, g_ins, g_del = Levenshtein(ref_words, goog_words)
            g_wer_lst += [g_wer]
            print(f"[{speaker}] Google {i} {g_wer:.3f} S:{g_sub}, I:{g_ins}, D:{g_del}")

    return k_wer_lst, g_wer_lst


def main(data_dir):
    k_wer_lst, g_wer_lst = [], []

    for speaker in get_speaker_names(data_dir):
        k_wers, g_wers = evaluate_wer_performance(data_dir, speaker)

        k_wer_lst += k_wers
        g_wer_lst += g_wers

    print("\n")
    print(" ".join([
        f"Kaldi WER Avg: {np.average(k_wer_lst):.4f}",
        f"Kaldi WER StD: {np.std(k_wer_lst):.4f}",
        f"Google WER Avg: {np.average(g_wer_lst):.4f}",
        f"Google WER StD: {np.std(g_wer_lst):.4f}"
    ]))


# endregion

if __name__ == "__main__":
    main(dataDir)
