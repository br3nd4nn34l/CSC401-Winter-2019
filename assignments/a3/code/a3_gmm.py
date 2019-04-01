from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = '/u/cs401/A3/data/'


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


# region Tensor Standardization Functions


def is_vector(x):
    """
    Return true if x is a vector, i.e.
    Only one axis in x is not length 1.
    """
    return max(x.shape) == np.product(x.shape)


def reshape_x_to_X(x, theta):
    """
    If x is a d-vector
        Reshape to (1, 1, d)
    If x is a (T, d)-matrix
        Reshape to (T, 1, d)
    If x is a 3D tensor, checks if it is of shape (T, 1, d)
    """

    d = theta.mu.shape[1]
    squeezed = np.squeeze(x)

    assert squeezed.shape in [1, 2, 3]

    if len(squeezed.shape) == 1:
        assert squeezed.shape[0] == d
        T = 1

    elif len(squeezed.shape) == 2:
        assert squeezed.shape[1] == d
        T = squeezed.shape[0]

    elif len(squeezed.shape) == 3:
        assert squeezed.shape[1:] == (1, d)
        return squeezed

    return x.reshape((T, 1, d))


def reshape_m_to_categories(m, theta):
    """
    If m is a number
        Reshape to array of size (1,)
    If M is a vector of size k
        Reshape to (k,)
    """
    squeezed = np.squeeze(m)

    # Number, reshape to (1,)
    if squeezed.shape == ():
        return squeezed.reshape((1,))

    # Must be a vector
    assert len(squeezed.shape) == 1

    # Every element must be less than number of categories
    M = theta.omega.shape[0]
    assert np.any(squeezed > M)

    return squeezed


# endregion


# region Log BMX Functions

def sum_xn2_over_smn(X, categories, theta):
    """
    :param X: array of shape (T, 1, d)
    :param categories: array of category indices
    :param theta: the model

    Let k = length of categories
    Returns k-vector where each element is aligned to categories.

    m-th element is equal to:
    Sum(n=1->d){ x[n]^2 / sig_m[n] }
    """

    # (t, 1, n)-th element is x_t[n]^2
    numerators = X ** 2  # Shape (T, 1, d)

    # (m, n)-th element is sig_m[n]
    denominators = theta.Sigma[categories]  # Shape (k, d)

    # (t, m, n)-th element is x_t[n]^2 / sig_m[n]
    terms = numerators / denominators  # Shape: (T, 1, d) / (_, k, d) -> (T, k, d)

    # Axis 0: sample
    # Axis 1: category
    # Axis 2: dimension
    # Want to sum along dimension
    return np.sum(terms, axis=2)  # Shape: (T, k)


def sum_xn_umn_over_smn(X, categories, theta):
    """
    :param X: array of shape (T, 1, d)
    :param categories: array of category indices
    :param theta: the model

    Let k = length of categories
    Returns k-vector where each element is aligned to categories.

    m-th element is equal to:
    Sum(n=1->d){ x[n] * mu_m[n] / sig_m[n] }
    """
    # (t, 1, n)-th element is x_t[n]
    xtn_arr = X  # Shape (T, 1, d)

    # (m, n)-th element is mu_m[n]
    mu_m_arr = theta.mu[categories]  # Shape (k, d)

    # (t, m, n)-th element is x_t[n] * mu_m[n]
    numerators = xtn_arr * mu_m_arr  # Shape: (T, 1, d) * (_, k, d) -> (T, k, d)

    # (m, n)-th element is sig_m[n]
    denominators = theta.Sigma[categories]  # Shape (k, d)

    # (t, m, n)-th element is (x_t[n] * mu_m[n]) / sig_m[n]
    terms = numerators / denominators  # Shape: (T, k, d) * (_, k, d) -> (T, k, d)

    # Axis 0: sample
    # Axis 1: category
    # Axis 2: dimension
    # Want to sum along dimension
    return np.sum(terms, axis=2)  # Shape (T, k)


def sum_umn2_over_smn(categories, theta):
    """
    :param categories: array of category indices
    :param theta: the model

    Let k = length of categories
    Returns k-vector where each element is aligned to categories.

    m-th element is equal to:
    Sum(n=1->d){ mu_m[n]^2 / sig_m[n] }
    """
    # (m, n)-th term is mu_m[n]^2
    numerators = theta.mu[categories] ** 2  # Shape: (k, d)

    # (m, n)-th term is sig_m[n]
    denominators = theta.Sigma[categories]  # Shape: (k, d)

    # (m, n)-th term is mu_m[n]^2 / sig_m[n]
    terms = numerators / denominators  # Shape: (k, d)

    # Axis 0: category
    # Axis 1: dimension
    # Want to sum along dimension
    return np.sum(terms, axis=1)  # Shape (k,)


def sum_log_smn(categories, theta):
    """
    :param categories: array of category indices
    :param theta: the model

    Let k = length of categories
    Returns k-vector where each element is aligned to categories.

    m-th element is equal to:
    Sum(n=1->d){ log(sig_m[n]) }
    """
    # (m, n)-th term is sig_m[n]
    sig_m = theta.Sigma[categories]  # Shape: (k, d)

    # (m, n)-th term is log(sig_m[n])
    log_sig_m = np.log(sig_m)  # Shape: (k, d)

    # Axis 0: category
    # Axis 1: dimension
    # Want to sum along dimension
    return np.sum(log_sig_m, axis=1)  # Shape: (k,)


def m_term_of_log_bmx(categories, theta):
    """
    :param categories: array of category indices
    :param theta: the model

    Let k = length of categories
    Returns k-vector where each element is aligned to categories.

    m-th element is equal to:
    Sum(n=1->d){ mu_m[n]^2 / sig_m[n] } +
    Sum(n=1->d){ log(sig_m[n]) } +
    d * log(2pi)
    """

    d = theta.mu.shape[1]
    d_log_2pi = d * np.log(2 * np.pi)

    # Shape: (k,)
    return sum_umn2_over_smn(categories, theta) + \
           sum_log_smn(categories, theta) + \
           d_log_2pi


def x_term_of_log_bmx(X, categories, theta):
    """
    :param X: array of shape (T, 1, d)
    :param categories: array of category indices
    :param theta: the model

    Let k = length of categories
    Returns k-vector where each element is aligned to categories.

    m-th element is equal to:
    Sum(n=1->d){ x[n]^2 / sig_m[n] } -
    2 * Sum(n=1->d){ x[n] * mu_m[n] / sig_m[n] }
    """
    # Shape (T, k)
    return sum_xn2_over_smn(X, categories, theta) - \
           (2 * sum_xn_umn_over_smn(X, categories, theta))


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    '''
    Returns the log probability of d-dimensional vector x using only component m of model myTheta
    See equation 1 of the handout

    As you'll see in tutorial, for efficiency, you can precompute something
    for 'm' that applies to all x outside of this function.
    If you do this, you pass that precomputed component in preComputedForM
    '''

    # Standardize input tensors
    X = reshape_x_to_X(x, myTheta)
    categories = reshape_m_to_categories(m, myTheta)

    # Term that depends on m
    if preComputedForM != []:
        m_term = preComputedForM
    else:
        m_term = m_term_of_log_bmx(categories, myTheta)  # Shape: (k,)

    # Term that depends on x
    x_term = x_term_of_log_bmx(X, categories, myTheta)  # Shape: (T, k)

    # Tensor to return
    return -0.5 * (x_term + m_term)  # Shape: (T, k)


# endregion

# region Log PMX Functions

def log_sum_wk_bkx(omega_all_cats, log_b_xt_all_cats):
    """
    :param omega_all_cats: array of shape (M,)

    :param log_b_xt_all_cats: array of shape (T, M)
    (log bmx for each x_t, for all m from 1 to M)

    Returns array of size (T,)
    Where t-th element is:
    log[Sum(k=1->M){ omega_k * b_k(xt) )}]
    """

    log_wk = np.log(omega_all_cats)  # Shape (M,)

    # logsum(w_k * b_k)
    # = logsum(exp(log(w_k * b_k)))
    # = logsum(exp(log(w_k) + log(b_k)))
    # = logsumexp(log(w_k) + log(b_k))
    terms = log_wk + log_b_xt_all_cats  # Shape: (M,) + (T, M) -> (T, M)

    # Axis 0: samples
    # Axis 1: categories
    # Want to sum along categories
    return logsumexp(terms, axis=1)


def log_wm(categories, theta):
    """
    :param categories: given categories
    :param theta: the model containing omega

    For k-vector of categories,
    returns k-vector where m-th element is:

    log(omega_m)
    """
    omega_m = theta.omega[categories]
    return np.log(omega_m)  # Shape: (k,)


def log_p_m_x(m, x, myTheta):
    '''
    Returns the log probability of the
    m^{th} component given d-dimensional vector x, and model myTheta
    See equation 2 of handout
    '''

    # All category numbers
    M = myTheta.omega.shape[0]
    all_cats = reshape_m_to_categories(np.arange(M), myTheta)  # Shape (M,)

    # Given category numbers
    these_cats = reshape_m_to_categories(m, myTheta)  # Shape (k,)

    # Precomputed m-terms for categories
    pre_all_cats = m_term_of_log_bmx(all_cats, myTheta)  # Shape (M,)
    pre_these_cats = pre_all_cats[these_cats]  # Shape (k,)

    # Data array
    X = reshape_x_to_X(x, myTheta)  # Shape (T, 1, d)

    # log_bmx for categories
    log_bmx_all_cats = log_b_m_x(all_cats, X, myTheta, pre_all_cats)  # Shape (T, M)

    # Shape: (k,) + (T, k) + (T, ) -> (T, k)
    return log_wm(these_cats, theta) + \
           log_b_m_x(these_cats, X, myTheta, pre_these_cats) + \
           log_sum_wk_bkx(myTheta.omega, log_bmx_all_cats)


# endregion

# region LogLik Functions

def logLik(log_Bs, myTheta):
    '''
    Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

    X can be training data, when used in train( ... ), and
    X can be testing data, when used in test( ... ).

    We don't actually pass X directly to the function because we instead pass:

    log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

    See equation 3 of the handout
    '''

    # Given shape (M, T) -> Want (T, M)
    log_b_xt_all_cats = log_Bs.transpose()

    # t-th element is log(p(x_t ; theta_s)
    # Shape: (T,)
    log_p_xt_theta_s = log_sum_wk_bkx(
        omega_all_cats=myTheta.omega,  # Shape: (M,)
        log_b_xt_all_cats=log_b_xt_all_cats  # Shape: (T, M)
    )

    # Sum across all t=1->T
    return np.sum(log_p_xt_theta_s)


# endregion

def compute_new_omega(log_pmxs):
    """
    Computes:
    w_m = Sum(t=1->T){ p(m|x_t;theta) / T}

    :param log_pmxs: (T, M) tensor
    (t,m)-th element is log(p(m|x_t;theta))
    """
    # For pmx's
    # Axis 0: Samples
    # Axis 1: Categories
    # Want to sum along samples
    log_sum_pmx = logsumexp(log_pmxs, axis=0) # Shape (M,)
    sum_pmx = np.exp(log_sum_pmx)

    # Grab T (samples axis)
    T = log_pmxs.shape[0]

    return sum_pmx / T


def compute_new_mu(log_pmxs, X):
    """
    Computes:
    mu_m = Sum(t=1->T){ p(m|x_t;theta) * x_t } /
            Sum(t=1->T){ (p(m|x_t;theta) }

    :param log_pmxs: (T, M) tensor
    (t,m)-th element is log(p(m|x_t;theta))
    """
    # For log_pmx's
    # Axis 0: Samples
    # Axis 1: Categories


    # Denominator: Want to sum along samples
    denominator = np.sum(log_pmxs, axis=0) # Shape (M,)

    # Grab T (samples axis)
    T = log_pmxs.shape[0]

    return total / T


def compute_new_sigma(log_pmxs, X, new_mu):
    pass


def compute_results(theta, X):
    """
    Computes the results needed to train the model

    :param theta: the model

    :param X: the data matrix, shape (T, M)

    :return:
    """
    M = theta.omega.shape[0]

    # List of all categories
    categories = np.arange(M)  # Shape: (M,)

    # Pre-computation for m
    m_pre_comp = m_term_of_log_bmx(categories, theta)  # Shape: (M,)

    # Shape (T, M): (t,m)-th element is log(b_m(x_t))
    log_Bs = log_b_m_x(categories, X, theta, m_pre_comp)

    # Shape (T, M): (t,m)-th element is log(p(m|x_t;theta))
    log_pmxs = log_p_m_x(categories, X, theta)

    # Shape: Scalar
    likelihood = np.exp(logLik(
        log_Bs.transpose(),  # logLik is Expecting (M,T)
        theta
    ))

    # Shape: (M,)
    new_omega = compute_new_omega(log_pmxs)

    # Shape: (M,d)
    new_mu = compute_new_mu(log_pmxs, X)

    # Shape: (M,d)
    new_sigma = compute_new_sigma(log_pmxs, X, new_mu)

    return (likelihood, new_omega, new_mu, new_sigma)


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    '''
    Train a model for the given speaker.
    Returns the theta (omega, mu, sigma)
    '''
    myTheta = theta(speaker, M, X.shape[1])

    categories = np.arange(M)  # Shape: (M,)

    prev_loss = -float('inf')
    improvement = float('inf')

    for i in range(maxIter):
        m_pre_comp = m_term_of_log_bmx(categories, myTheta)  # Shape: (M,)

        log_Bs = log_b_m_x(categories, X, myTheta, m_pre_comp)  # Shape: (T, M)
        log_pmxs = log_p_m_x(categories, X, myTheta)  # Shape: (T, M)

    print('TODO')
    return myTheta


def test(mfcc, correctID, models, k=5):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    print('TODO')
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate 
    numCorrect = 0;
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
