from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, chi2

import numpy as np
import argparse
import sys
import os

import csv
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix

# region Constants / Helpers


# Note: .fit() will reset each classifier's params
# https://stackoverflow.com/questions/51613004/scikit-learn-forget-previous-train-data
classifiers = [

    # Linear SVC
    SVC(kernel="linear", max_iter=1000,
        random_state=401),

    # Radial SVC, Gamma=2
    SVC(kernel="rbf", gamma=2, max_iter=1000,
        random_state=401),

    # Random Forest Classifier
    RandomForestClassifier(n_estimators=10, max_depth=5,
                           random_state=401),

    # MLPC Classifier
    MLPClassifier(alpha=0.05,
                  random_state=401),

    # AdaBoost Classifier
    AdaBoostClassifier(random_state=401),
]

def write_csv(filename, rows):
    """
    TODO DOCSTRING
    :param filename:
    :param rows:
    :return:
    """
    with open(filename, mode="w", newline="") as csv_ptr:
        csv_writer = csv.writer(csv_ptr)
        for row in rows:
            csv_writer.writerow(row)


# endregion

# region Metrics

def accuracy(C):
    '''
    Compute accuracy given Numpy array confusion matrix C.
    Returns a floating point value
    '''

    # Sum of diagonal (number of i's correctly classified as i's)
    numerator = np.diag(C).sum()

    # Sum of everything (number of samples)
    denominator = C.sum()

    return numerator / denominator


def recall(C):
    '''
    Compute recall given Numpy array confusion matrix C.
    Returns a list of floating point values
    '''

    # Numerators: diagonal (i's correctly classified as i's)
    numerators = np.diag(C)

    # Denominators: sum along columns (number of samples classified as i)
    denominators = C.sum(axis=1)

    return numerators / denominators


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C.
    Returns a list of floating point values '''

    # Numerators: diagonal (i's correctly classified as i's)
    numerators = np.diag(C)

    # Denominators: sum along rows (number of samples classified as i)
    denominators = C.sum(axis=0)

    return numerators / denominators


# endregion

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''

    # Grab data matrix from NPZ dictionary (assumed to be only item)
    data_matrix = dict(np.load(filename)).popitem()[1]

    # Split the data into train (80%) and test (20%),
    train, test = train_test_split(data_matrix,
                                   random_state=401,  # Seed RNG for consistency
                                   train_size=0.8,
                                   test_size=0.2)

    # All columns except last are X, last column is Y
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    # CSV rows
    csv_rows = []

    for (i, c) in enumerate(classifiers):
        # Train classifier
        c.fit(X_train, y_train)

        # Determine confusion matrix
        confusion = confusion_matrix(y_test,
                                     c.predict(X_test))

        # Add the row to the CSV
        csv_rows += [[
            i + 1,  # Number of classifier

            # Accuracy, recall and precision of classifier
            accuracy(confusion),
            *recall(confusion).flatten(),
            *precision(confusion).flatten(),

            # Row-by-row (flattened) of confusion matrix
            *confusion.flatten()
        ]]

    # Write the csv
    write_csv("a1_3.1.csv", csv_rows)

    # Index of the best classifier (accuracy is second column in csv rows)
    iBest = max(
        range(len(csv_rows)),
        key=lambda ind: csv_rows[ind][1]
    )

    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    '''

    best_classifier = classifiers[iBest]

    accuracies = []

    for num_data in [1000, 5000, 10000, 15000, 20000]:
        X_train_subset = X_train[:num_data]
        y_train_subset = y_train[:num_data]

        # Fit will reset the classifier to suit the new data
        best_classifier.fit(X_train_subset, y_train_subset)

        # Score method returns accuracy
        # (3.1 already demonstrated me using confusion matrix to do this)
        accuracies += [best_classifier.score(X_test, y_test)]

    # Analysis
    analysis = [
        "I expected the classifier to perform with higher accuracy "
        "as it was given more training data, with diminishing "
        "returns to performance (i.e. a negative second derivative "
        "of accuracy WRT training size). "
        "This was indeed the case, as marginal accuracies decreased with more data "
        "(0.048125->0.00925->0.007875->0.001875)."

        "The diminishing accuracy returns WRT training size is "
        "probably due to the model reaching the peak of its "
        "representational power. "

        "In other words, the methodology that the model uses "
        "to make conclusions is inherently limited and "
        "cannot properly emulate the underlying distribution. "
        "Using more data to tune the parameters of such a model "
        "will only result in small accuracy gains. "
    ]

    # Write the csv rows
    write_csv("a1_3.2.csv", [accuracies, analysis])

    X_1k, y_1k = X_train[:1000], y_train[:1000]

    return (X_1k, y_1k)


def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''

    best_classifier = classifiers[i]

    csv_rows = []

    # For 32K training set, report p-values for each k
    for num_features in [5, 10, 20, 30, 40, 50]:

        selector = SelectKBest(f_classif, k=num_features)
        selector.fit(X_train, y_train)

        csv_rows += [[num_features, *selector.pvalues_]]

    # Want accuracy for k=5, 1k and 32k training cases
    accuracies = []
    for (train_x, train_y) in [(X_1k, y_1k), (X_train, y_train)]:
        selector = SelectKBest(f_classif, k=5)

        train_x_reduced = selector.fit_transform(train_x, train_y)
        best_classifier.fit(train_x_reduced, train_y)

        test_x_reduced = selector.transform(X_test)

        accuracies += [best_classifier.score(test_x_reduced, y_test)]

    csv_rows += [accuracies]

    # Line 8: Q3.3.3.a: data size vs. chosen features, explanation TODO


    # Line 9: Q3.3.3.b: data size vs. p-values, explanation TODO

    # Line 10: Q3.3.3.c: name top 5 features for 32K.
    # Explain why they differentiate classes TODO


    # Write the CSV
    write_csv("a1_3.3.csv", csv_rows)

def class34(filename, i):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')


def main(args):
    """
    TODO DOCSTRING
    :param args:
    :return:
    """
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)

    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Per')

    parser.add_argument("-i", "--input",
                        help="the input npz file from Task 2",
                        required=True)

    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    main(args)
