from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


import numpy as np
import argparse
import sys
import os

import csv

from scipy import stats


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


def load_data_matrix(filename):
    """
    TODO DOCSTRING
    :param filename:
    :return:
    """
    # Grab data matrix from NPZ dictionary (assumed to be only item)
    return dict(np.load(filename)).popitem()[1]


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
    data_matrix = load_data_matrix(filename)

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

    # Select the best classifier
    best_classifier = classifiers[i]

    # Dictionary used for analysis
    # {(num_features, train_size) : (accuracy, p_values, )}
    results = {}

    for num_features in [5, 10, 20, 30, 40, 50]:
        for (train_x, train_y) in [(X_1k, y_1k), (X_train, y_train)]:
            dimensionality = train_x.shape[1]
            train_size = train_y.shape[0]

            # Train the classifier
            selector = SelectKBest(f_classif, k=num_features)
            train_x_reduced = selector.fit_transform(train_x, train_y)
            best_classifier.fit(train_x_reduced, train_y)

            # Determine accuracy on test set
            test_x_reduced = selector.transform(X_test)
            acc = best_classifier.score(test_x_reduced, y_test)

            # Determine attributes selected
            attr_nums = np.arange(dimensionality).reshape(1, -1) + 1
            selected_attrs = selector.transform(attr_nums)

            results[(num_features, train_size)] = [
                acc,
                selector.pvalues_,
                selected_attrs
            ]

    # Gather results for report
    csv_rows = []

    # For 32K training set, report p-values for each k
    for num_features in [5, 10, 20, 30, 40, 50]:
        p_values = results[(num_features, 32000)][1]
        csv_rows += [[num_features, *p_values]]

    # Want accuracy for k=5, 1k and 32k training cases
    csv_rows += [[
        results[(5, train_size)][0]
        for train_size in [1000, 32000]
    ]]

    # Line 8: Q3.3.3.a: common features, explanation TODO
    print("Data Size vs Top 5 Chosen Features")
    for train_size in [1000, 32000]:
        features = results[(5, train_size)][2]
        print(train_size, features)
    print("")
    csv_rows += [[
        "For the 1K and 32K training data, the "
        "common features are #adverbs (11), "
        "and receptiviti_intellectual (150). "
        "People with greater writing ability "
        "(indicative of greater education/intellect) are probably more "
        "likely to 'flesh-out' their communication with adverbs. "
        "Education and intelligence have been shown to "
        "correlate with political alignment, which probably "
        "makes these features good indicators, "
        "regardless of training data, "
        "for the classifier. "
        "Article that summarizes various studies on intelligence vs. politics: "
        "https://www.psychologytoday.com/ca/blog/unique-everybody-else/201305/intelligence-and-politics-have-complex-relationship"
    ]]

    # Line 9: Q3.3.3.b: data size vs. p-values, explanation TODO
    print("Data Size vs. Avg P-Value")
    for train_size in [1000, 32000]:

        p_value_sum = 0
        num_p_values = 0

        for num_features in [5, 10, 20, 30, 40, 50]:
            # Grab p_values, exclude NANs
            p_values = results[(num_features, train_size)][1]
            p_values = p_values[~np.isnan(p_values)]

            # Increment sums
            p_value_sum += np.sum(p_values)
            num_p_values += p_values.size

        avg_p_val = p_value_sum / num_p_values
        print(train_size, avg_p_val)
    print("")
    csv_rows += [[
        "P-Values are generally higher for less training data "
        "(average p-value is 0.17 for 1K training samples, "
        "and 0.019 for 32K training samples). "
        "The p-value of a feature is inversely proportional "
        "to the predictive power of that feature. "
        "A classifier with more training data has more "
        "information to decide the predictive "
        "power of various features. "
        "Therefore, it makes sense that more "
        "training data would produce lower p-values "
        "(features are more predictive with more training data)."
    ]]

    # Line 10: Q3.3.3.c: name top 5 features for 32K,
    # explain how they serve as class differentiators.
    print("Top 5 features for 32K")
    print(results[(5, 32000)][2])

    print("Class Averages for Top 5 Features (listed above)")
    for class_num in [0, 1, 2, 3]:
        class_rows = X_train[y_train == class_num]
        avg_row = np.average(class_rows, axis=0)
        print(class_num, " ".join(str(avg_row[x]) for x in [10, 83, 96, 143, 149]))
    csv_rows += [[
        "The top 5 features for 32K training data are "
        "11, 84, 97, 144 and 150 (1-indexed). "

        "These correspond to #adverbs (11), liwc_motion (84), liwc_relativ (97), "
        "receptiviti_health_oriented (144), receptiviti_intellectual (150). "

        "Number of Adverbs (11), liwc_relativ (97), and "
        "receptiviti_intellectual (150) probably "
        "correlate with education/intelligence. "
        "As discussed in 3.3.3.a, education and "
        "intelligence correlate with political alignment. "

        "liwc_motion (84) could indicate discussion about "
        "(political) movement, change and transformation. "
        "This could indicate a person's feelings towards "
        "the status-quo of society, and therefore their political alignment. "

        "receptiviti_health_oriented (144) could indicate discussion on healthcare "
        "and human-welfare isssues, which are important issues in left-leaning politics. "
    ]]

    # Write the CSV
    write_csv("a1_3.3.csv", csv_rows)


def class34(filename, i):
    ''' This function performs experiment 3.4
    
    Parameters
    filename : string, the name of the npz file from Task 2
    i: int, the index of the supposed best classifier (from task 3.1)
    '''

    # Load data matrix, split into X, y
    data_matrix = load_data_matrix(filename)
    X, y = data_matrix[:, :-1], data_matrix[:, -1]

    # (i, j): j-th classifier's accuracy on the i-th fold of data
    accuracies = []

    # K-fold the data with k=5, train and test each classifier on each fold
    kf = KFold(n_splits=5, shuffle=True, random_state=401)
    for (train_indices, test_indices) in kf.split(X):
        x_train, y_train = X[train_indices], y[train_indices]
        x_test, y_test = X[test_indices], y[test_indices]

        # Record classifier's accuracy (on this particular fold)
        accuracies += [[
            c.fit(x_train, y_train)
                .score(x_test, y_test)
            for c in classifiers
        ]]

    # Take transpose of accuracies such that each
    # row corresponds to each classifier's accuracies
    class_to_accs = np.array(accuracies).transpose()

    # Determine p-values using t-test
    # (compare classifiers against supposed best)
    p_values = [
        stats.ttest_rel(class_to_accs[i], class_to_accs[j]).pvalue
        for j in range(len(classifiers))
            if j != i
    ]


    analysis = [
        "The best classifier (#5, AdaBoost, 3.1 accuracy of 0.459) "
        "had p-values of ~2E-7, ~2E-6, ~4E-4, and ~8E-3, "
        "against the following classifiers: "
        "Linear SVC, Radial SVC, Random Forest, Multi-Layer-Perceptron."
        "The relatively small p-values listed are highly "
        "statistically significant with all P < 0.001 - "
        "there is less than a 1 in 1000 chance that "
        "AdaBoost performs same as the competing classifiers."
    ]

    # CSV rows: accuracies, p_values, analysis
    write_csv("a1_3.4.csv", accuracies + [p_values] + [analysis])


def main(args):
    """
    TODO DOCSTRING
    :param args:
    :return:
    """
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)

    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.input, iBest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Per')

    parser.add_argument("-i", "--input",
                        help="the input npz file from Task 2",
                        required=True)

    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    main(args)
