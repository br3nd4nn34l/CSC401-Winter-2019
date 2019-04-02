from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

# Seed for consistency
random.seed(401)
np.random.seed(401)

dataDir = '/u/cs401/A3/data/'

# TODO CHANGE FOR RUNNING ON CDF
dataDir = '../data/'


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


# region Tensor Standardization Functions

def reshape_x_to_X(x, model):
    """
    If x is a d-vector
        Reshape to (1, 1, d)
    If x is a (T, d)-matrix
        Reshape to (T, 1, d)
    If x is a 3D tensor, checks if it is of shape (T, 1, d)
    """

    d = model.mu.shape[1]
    squeezed = np.squeeze(x)

    assert len(squeezed.shape) in [1, 2, 3]

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


def reshape_m_to_components(m, model):
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

    # Every element must be less than number of components
    M = model.omega.shape[0]
    assert np.all(squeezed < M)

    return squeezed


# endregion

# region Log BMX Functions


def sum_xn2_over_smn(X, components, model):
    """
    :param X: array of shape (T, 1, d)
    :param components: array of component indices
    :param model: the model

    Let k = length of components
    Let k = length of components
    Returns (T,k) tensor where (t, m)-th element is equal to:
    Sum(n=1->d){ x[n]^2 / sig_m[n] }
    """

    # (t, 1, n)-th element is x_t[n]^2
    numerators = X ** 2  # Shape (T, 1, d)

    # (m, n)-th element is sig_m[n]
    denominators = model.Sigma[components]  # Shape (k, d)

    # (t, m, n)-th element is x_t[n]^2 / sig_m[n]
    terms = numerators / denominators  # Shape: (T, 1, d) / (_, k, d) -> (T, k, d)

    # Axis 0: sample
    # Axis 1: component
    # Axis 2: dimension
    # Want to sum along dimension
    return np.sum(terms, axis=2)  # Shape: (T, k)


def sum_xn_umn_over_smn(X, components, model):
    """
    :param X: array of shape (T, 1, d)
    :param components: array of component indices
    :param model: the model

    Let k = length of components
    Returns (T,k) tensor where (t, m)-th element is equal to:
    Sum(n=1->d){ x_t[n] * mu_m[n] / sig_m[n] }
    """
    # (t, 1, n)-th element is x_t[n]
    xtn_arr = X  # Shape (T, 1, d)

    # (m, n)-th element is mu_m[n]
    mu_m_arr = model.mu[components]  # Shape (k, d)

    # (t, m, n)-th element is x_t[n] * mu_m[n]
    numerators = xtn_arr * mu_m_arr  # Shape: (T, 1, d) * (_, k, d) -> (T, k, d)

    # (m, n)-th element is sig_m[n]
    denominators = model.Sigma[components]  # Shape (k, d)

    # (t, m, n)-th element is (x_t[n] * mu_m[n]) / sig_m[n]
    terms = numerators / denominators  # Shape: (T, k, d) * (_, k, d) -> (T, k, d)

    # Axis 0: sample
    # Axis 1: component
    # Axis 2: dimension
    # Want to sum along dimension
    return np.sum(terms, axis=2)  # Shape (T, k)


def sum_umn2_over_smn(components, model):
    """
    :param components: array of component indices
    :param model: the model

    Let k = length of components
    Returns k-vector where each element is aligned to components.

    m-th element is equal to:
    Sum(n=1->d){ mu_m[n]^2 / sig_m[n] }
    """
    # (m, n)-th term is mu_m[n]^2
    numerators = model.mu[components] ** 2  # Shape: (k, d)

    # (m, n)-th term is sig_m[n]
    denominators = model.Sigma[components]  # Shape: (k, d)

    # (m, n)-th term is mu_m[n]^2 / sig_m[n]
    terms = numerators / denominators  # Shape: (k, d)

    # Axis 0: component
    # Axis 1: dimension
    # Want to sum along dimension
    return np.sum(terms, axis=1)  # Shape (k,)


def sum_log_smn(components, model):
    """
    :param components: array of component indices
    :param model: the model

    Let k = length of components
    Returns k-vector where each element is aligned to components.

    m-th element is equal to:
    Sum(n=1->d){ log(sig_m[n]) }
    """
    # (m, n)-th term is sig_m[n]
    sig_m = model.Sigma[components]  # Shape: (k, d)

    # (m, n)-th term is log(sig_m[n])
    log_sig_m = np.log(sig_m)  # Shape: (k, d)

    # Axis 0: component
    # Axis 1: dimension
    # Want to sum along dimension
    return np.sum(log_sig_m, axis=1)  # Shape: (k,)


def m_term_of_log_bmx(components, model):
    """
    :param components: array of component indices
    :param model: the model

    Let k = length of components
    Returns k-vector where each element is aligned to components.

    m-th element is equal to:
    Sum(n=1->d){ mu_m[n]^2 / sig_m[n] } +
    Sum(n=1->d){ log(sig_m[n]) } +
    d * log(2pi)
    """

    d = model.mu.shape[1]
    d_log_2pi = d * np.log(2 * np.pi)

    # Shape: (k,)
    return sum_umn2_over_smn(components, model) + \
           sum_log_smn(components, model) + \
           d_log_2pi


def x_term_of_log_bmx(X, components, model):
    """
    :param X: array of shape (T, 1, d)
    :param components: array of component indices
    :param model: the model

    Let k = length of components
    Returns k-vector where each element is aligned to components.

    m-th element is equal to:
    Sum(n=1->d){ x[n]^2 / sig_m[n] } -
    2 * Sum(n=1->d){ x[n] * mu_m[n] / sig_m[n] }
    """
    # Shape (T, k)
    return sum_xn2_over_smn(X, components, model) - \
           (2 * sum_xn_umn_over_smn(X, components, model))


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
    components = reshape_m_to_components(m, myTheta)

    # Term that depends on m
    if preComputedForM != []:
        m_term = preComputedForM
    else:
        m_term = m_term_of_log_bmx(components, myTheta)  # Shape: (k,)

    # Term that depends on x
    x_term = x_term_of_log_bmx(X, components, myTheta)  # Shape: (T, k)

    # Tensor to return
    return -0.5 * (x_term + m_term)  # Shape: (T, k)


# endregion

# region Log PMX Functions

def log_sum_wk_bkx(omega_all_comps, log_b_xt_all_comps):
    """
    :param omega_all_comps: array of shape (M,)

    :param log_b_xt_all_comps: array of shape (T, M)
    (log bmx for each x_t, for all m from 1 to M)

    Returns array of size (T,)
    Where t-th element is:
    log[Sum(k=1->M){ omega_k * b_k(xt) )}]
    """

    log_wk = np.log(omega_all_comps).flatten()  # Shape (M,)

    # logsum(w_k * b_k)
    # = logsum(exp(log(w_k * b_k)))
    # = logsum(exp(log(w_k) + log(b_k)))
    # = logsumexp(log(w_k) + log(b_k))
    terms = log_wk + log_b_xt_all_comps  # Shape: (M,) + (T, M) -> (T, M)

    # Axis 0: samples
    # Axis 1: components
    # Want to sum along components
    return logsumexp(terms, axis=1)


def log_wm(components, model):
    """
    :param components: given components
    :param model: the model containing omega

    For k-vector of components,
    returns k-vector where m-th element is:

    log(omega_m)
    """
    omega_m = model.omega[components].flatten()
    return np.log(omega_m)  # Shape: (k,)


def log_p_m_x(m, x, myTheta):
    '''
    Returns the log probability of the
    m^{th} component given d-dimensional vector x, and model myTheta
    See equation 2 of handout
    '''
    # All component numbers
    M = myTheta.omega.shape[0]
    all_comps = reshape_m_to_components(np.arange(M), myTheta)  # Shape (M,)

    # Given component numbers
    these_comps = reshape_m_to_components(m, myTheta)  # Shape (k,)

    # Precomputed m-terms for components
    pre_all_comps = m_term_of_log_bmx(all_comps, myTheta)  # Shape (M,)
    pre_these_comps = pre_all_comps[these_comps]  # Shape (k,)

    # Data array
    X = reshape_x_to_X(x, myTheta)  # Shape (T, 1, d)

    # log_bmx for components
    log_bmx_all_comps = log_b_m_x(all_comps, X, myTheta, pre_all_comps)  # Shape (T, M)

    # Shape: (k,) + (T, k) - (T, 1) -> (T, k)
    return log_wm(these_comps, myTheta) + \
           log_b_m_x(these_comps, X, myTheta, pre_these_comps) - \
           log_sum_wk_bkx(myTheta.omega, log_bmx_all_comps)[:, np.newaxis]


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
    log_b_xt_all_comps = log_Bs.transpose()

    # t-th element is log(p(x_t ; theta_s)
    # Shape: (T,)
    log_p_xt_theta = log_sum_wk_bkx(
        omega_all_comps=myTheta.omega,  # Shape: (M,)
        log_b_xt_all_comps=log_b_xt_all_comps  # Shape: (T, M)
    )

    # Sum across all t=1->T
    return np.sum(log_p_xt_theta)


# endregion

# region Training Functions

def compute_log_sum_pmx(log_pmxs):
    """
    :param log_pmxs: (T, M) tensor
    (t,m)-th element is log(p(m|x_t;theta))

    Returns (M,) tensor where m-th element equals:
    log(Sum(t=1->T){ p(m|x_t;theta) })
    """
    # For log_pmx's
    # Axis 0: Samples
    # Axis 1: components
    # Want to sum along samples
    return logsumexp(log_pmxs, axis=0)  # Shape (T, M) -> (M,)


def compute_new_omega(log_pmxs):
    """
    :param log_pmxs: (T, M) tensor
    (t,m)-th element is log(p(m|x_t;theta))

    Returns (M,1) tensor where m-th element equals:
    omega_m = Sum(t=1->T){ p(m|x_t;theta) / T}
    """
    # m-th element is Sum(t=1->T){ p(m|x_t;theta) }
    log_sum_pmx = compute_log_sum_pmx(log_pmxs)  # Shape (M,)

    # Log T
    log_T = np.log(log_pmxs.shape[0])

    # (log of division) is (subtraction of log)
    return np.exp(log_sum_pmx - log_T)[:, np.newaxis]  # Shape (M,1)


def compute_new_mu(log_pmxs, X_reshaped):
    """
    :param log_pmxs: (T, M) tensor
    (t,m)-th element is log(p(m|x_t;theta))

    :param X_reshaped: (T, 1, d) data tensor
    (t, 1, d)-th element is (x_t[d])

    Returns (M,d) tensor where (m,d)-th element is:
    mu_m = Sum(t=1->T){ p(m|x_t;theta) * x_t[d] } /
            Sum(t=1->T){ (p(m|x_t;theta) }
    """
    # (t, m, 1)-th element is p(m|x_t;theta)
    pmx_reshape = np.exp(log_pmxs)[..., np.newaxis]  # Shape (T, M, 1)

    # (t, m, d)-th element: p(m|x_t;theta) * x_t[d]
    pmx_xt = pmx_reshape * X_reshaped  # Shape (T, M, 1) * (T, 1, d) -> (T, M, d)

    # Axis 0: Sample
    # Axis 1: component (m)
    # Axis 2: Dimension (d)
    # Want to sum along samples (axis 0)
    # (m, d)-th element: Sum(t=1->T){ p(m|x_t;theta) * x_t[d] }
    sum_pmx_xt = pmx_xt.sum(axis=0)

    # m-th element is log(Sum(t=1->T){ (p(m|x_t;theta) })
    log_sum_pmx = compute_log_sum_pmx(log_pmxs)  # Shape (M,)
    sum_pmx = np.exp(log_sum_pmx)  # Shape (M, )

    # Shape (M, d) / (M, 1) -> (M, d)
    return sum_pmx_xt / sum_pmx[:, np.newaxis]


def compute_new_sigma(log_pmxs, X_reshaped, new_mu):
    """
    :param log_pmxs: (T, M) tensor
    (t,m)-th element is log(p(m|x_t;theta))

    :param X_reshaped: (T, 1, d) data tensor
    (t, 1, d)-th element is (x_t[d])

    :param new_mu: (M, d) tensor where (m,d)-th element is:
    mu_m = Sum(t=1->T){ p(m|x_t;theta) * x_t[d] } /
            Sum(t=1->T){ (p(m|x_t;theta) }

    Returns (M, d) tensor where (m, d)-th element is:
    (
        Sum(t=1->T){ p(m|x_t;theta) * x_t^2[d] } /
        Sum(t=1->T){ (p(m|x_t;theta) }
    ) - (mu_m^2)[d]
    """

    # (t, m, 1)-th element is p(m|x_t;theta)
    pmx_reshape = np.exp(log_pmxs)[..., np.newaxis]  # Shape (T, M, 1)

    # (t, m, d)-th element: p(m|x_t;theta) * x_t^2[d]
    pmx_xt2 = pmx_reshape * (X_reshaped ** 2)  # Shape (T, M, 1) * (T, 1, d) -> (T, M, d)

    # Axis 0: Sample
    # Axis 1: component (m)
    # Axis 2: Dimension (d)
    # Want to sum along samples (axis 0)
    # (m, d)-th element: Sum(t=1->T){ p(m|x_t;theta) * x_t[d]^2 }
    sum_pmx_xt2 = pmx_xt2.sum(axis=0)

    # m-th element is log(Sum(t=1->T){ (p(m|x_t;theta) })
    log_sum_pmx = compute_log_sum_pmx(log_pmxs)  # Shape (M,)
    sum_pmx = np.exp(log_sum_pmx)  # Shape (M, )

    # The fraction of sums,
    fraction = sum_pmx_xt2 / sum_pmx[:, np.newaxis]  # Shape (M, d) / (M, 1) -> (M, d)

    # Shape (M, d) - (M, d) -> (M, d)
    return fraction - (new_mu ** 2)


def compute_results(model, X_reshaped):
    """
    Computes the results needed to train the model

    :param model: the model

    :param X_reshaped: the data tensor, shape (T, 1, d)

    :return:
    """
    M = model.omega.shape[0]

    # List of all components
    components = np.arange(M)  # Shape: (M,)

    # Pre-computation for m
    m_pre_comp = m_term_of_log_bmx(components, model)  # Shape: (M,)

    # Shape (T, M): (t,m)-th element is log(b_m(x_t))
    log_Bs = log_b_m_x(components, X_reshaped, model, m_pre_comp)

    # Shape (T, M): (t,m)-th element is log(p(m|x_t;model))
    log_pmxs = log_p_m_x(components, X_reshaped, model)

    log_likelihood = logLik(
        log_Bs.transpose(),  # logLik is Expecting (M,T)
        model
    )

    # Shape: (M,)
    new_omega = compute_new_omega(log_pmxs)

    # Shape: (M,d)
    new_mu = compute_new_mu(log_pmxs, X_reshaped)

    # Shape: (M,d)
    new_sigma = compute_new_sigma(log_pmxs, X_reshaped, new_mu)

    return (log_likelihood, new_omega, new_mu, new_sigma)


def initialize_theta(speaker, M, X):
    """

    Initialize theta as specified in slide 28 of A3 tutorial 1

    :param speaker: name of the speaker
    :param M: number of components for the model
    :param X: the (T, d) data matrix

    Returns the initialized theta.

    """
    T, d = X.shape
    ret = theta(speaker, M, d)

    # Initialize mu to random vectors from data
    rand_inds = np.random.randint(low=0, high=T, size=M)  # Shape (M,)
    ret.mu = X[rand_inds, :]  # Shape (T, d)[(M,)] -> (M, d)

    # Initialize sigma randomly
    ret.Sigma = np.random.rand(*ret.Sigma.shape)  # Between 0 and 1

    # Initialize omega randomly
    # Omegas must add up to one
    # Omegas must be between 0 and 1
    ret.omega = np.abs(np.random.rand(*ret.omega.shape))
    ret.omega /= ret.omega.sum()

    return ret


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    '''
    Train a model for the given speaker.
    Returns the theta (omega, mu, sigma)
    '''
    myTheta = initialize_theta(speaker, M, X)

    prev_log_lik = -float('inf')
    improvement = float('inf')

    X_reshaped = reshape_x_to_X(X, myTheta)

    # Repeat for specified number of iterations
    for i in range(maxIter + 1):  # Equivalent to a leq while-loop

        # Abort if improvement did not exceed epsilon
        if improvement < epsilon:
            break

        # Compute new parameters
        log_likelihood, new_omega, new_mu, new_sigma = \
            compute_results(myTheta, X_reshaped)

        # Update parameters
        myTheta.omega = new_omega
        myTheta.mu = new_mu
        myTheta.Sigma = new_sigma

        # Update improvement
        improvement = log_likelihood - prev_log_lik
        prev_log_lik = log_likelihood

    return myTheta


# endregion

def test(mfcc, correctID, models, k=5):
    '''
    Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
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

    def model_log_lik(mod):
        mfcc_reshaped = reshape_x_to_X(mfcc, mod)

        M = mod.omega.shape[0]

        # List of all components
        components = np.arange(M)  # Shape: (M,)

        # Pre-computation for m
        m_pre_comp = m_term_of_log_bmx(components, mod)  # Shape: (M,)

        log_Bs = log_b_m_x(components, mfcc_reshaped, mod, m_pre_comp)

        return logLik(
            log_Bs.transpose(),  # This function expects transpose
            myTheta=mod
        )

    # Models from best->worst
    best_to_worst_models = list(sorted(
        models, key=model_log_lik, reverse=True
    ))

    best_model = best_to_worst_models[0]
    top_k_models = best_to_worst_models[:k]

    correct_model = models[correctID]
    print(correct_model.name)
    for mod in top_k_models:
        print(f"{mod.name} {model_log_lik(mod)}")

    return 1 if (best_model.name == correct_model.name) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
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
    numCorrect = 0
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print(f"Final Accuracy: {accuracy}")
