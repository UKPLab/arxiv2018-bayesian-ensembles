import numpy as np
from scipy.special import psi # the Digamma function

def ibccvb(annotation_matrix, num_classes, class_proportion_smoothing=0.001, annotator_smoothing=0.1,
           annotator_accuracy_bias=0.9, last_corner_bias=1.0, max_iter=10, target_labels=None, uniform_priors=[], verbose=False):
    '''
    A simple implementation of the IBCC-VB method described in
    https://arxiv.org/abs/1206.1831

    IBCC is the name of the model (independent Bayesian classifier combination).
    VB (variational Bayes) is the learning technique. Previously, Gibbs' sampling was used but is slower
    with little gain in performance.

    It combines multiple classifications to produce a posterior probability distribution
    over class labels and estimate annotator informativeness.

    :param annotation_matrix: A matrix containing the labels from multiple annotators, where
    each row is a vector of labels for one data point, and each column contains the annos from one annotator.
    :param num_classes: number of target classes.
    :param class_proportion_smoothing: the method learns the proportion of each class label in the dataset.
    Increasing this value smooths the estimate of the class proportion to prevent overfitting
    with imbalanced datasets. Increase this value if the model is failing to predict some rare classes at all.
    :param annotator_smoothing:
    smooths the estimates of annotator reliability. This can be increased to prevent overfitting
    if some annotators have very small numbers of labels or if the model appears to be over-confident.
    :param annotator_accuracy_bias:
    This is a prior weight we give to annotators to assume that they are likely to be better than random.
    Set this to zero if you don't know whether the annotators are better than random, or set it to larger
    values to give a stronger a priori assumption that annotators are good.
    :param max_iter:
    The algorithm is iterative so this sets that maximum number of iterations in case it does not converge.
    :param target_labels:
    A 1-D numpy array of length N, where N is the number of data points. This is used to supply training labels for
    semi-supervised learning. The known trainin points should be set to their known values, and test points should be
    set to -1.
    :param uniform_priors:
    A list or np array containing indices of annotators for whom a uniform prior should be applied. Use this if you
    have no expectation that the annos will correlate with ground truth.
    :return t:
    The posterior distribution over the true labels. Each row corrsponds to one data point and sums to one.
    :return annotator_acc:
    The posterior estimate of annotator accuracy -- this is a vector containing a single value between 0
    and 1 for each annotator.
    '''
    eps = 1e-4

    N = annotation_matrix.shape[0] # number of data points
    num_annotators = annotation_matrix.shape[1]

    # t contains estimates of the posterior distribution over the true labels
    # init t using a proportions of votes for each class, treating annotators equally
    t = np.zeros((N, num_classes)) + class_proportion_smoothing
    for l in range(num_classes):
        t[:, l] += np.sum(annotation_matrix == l, axis=1)
    t /= np.sum(t, axis=1)[:, None]

    # put in any training labels
    if target_labels is not None:
        tridxs = np.argwhere(target_labels != -1).flatten()
        target_labels = np.array(target_labels).astype(int)
        t[tridxs] = 0
        t[tridxs, target_labels[tridxs]] = 1

    annotator_smoothing = annotator_smoothing * np.ones((num_classes, num_classes, num_annotators)) / float(num_classes-1)
    nonuniform_priors = np.ones(num_annotators, dtype=bool)
    if len(uniform_priors):
        nonuniform_priors[uniform_priors] = False

    for l in range(num_classes):
        annotator_smoothing[l, l, nonuniform_priors] *= num_classes-1
        annotator_smoothing[l, l, nonuniform_priors] += annotator_accuracy_bias

    annotator_smoothing[-1, -1, nonuniform_priors] *= last_corner_bias

    class_proportion_smoothing = class_proportion_smoothing * np.ones(num_classes)
    annotator_pseudocounts = np.copy(annotator_smoothing)

    for i in range(max_iter):
        class_pseudocounts = class_proportion_smoothing + np.sum(t, axis=0) # in the papers this is called \nu
        lnkappa = psi(class_pseudocounts) - psi(np.sum(class_pseudocounts)) # expected log of class proportions

        for l in range(num_classes):
            # in the papers this is referred to as \alpha
            annotator_pseudocounts[:, l, :] = annotator_smoothing[:, l, :] + t.T.dot(annotation_matrix == l)
        # expected log of the annotator confusion matrices
        lnpi = psi(annotator_pseudocounts) - psi(np.sum(annotator_pseudocounts, axis=1)[:, None, :])

        # posterior probability of each class label at each data point
        t_old = t
        t = lnkappa[None, :]
        for l in range(num_classes):
            t = (annotation_matrix == l).dot(lnpi[:, l, :].T) + t
        t = np.exp(t)
        t /= np.sum(t, axis=1)[:, None]

        if target_labels is not None:
            t[tridxs] = 0
            t[tridxs, target_labels[tridxs]] = 1

        if np.max(np.abs(t_old - t)) < eps:
            break

    # if verbose:
    print('Completed in %i iterations' % i)

    #print('Class proportions = %s' % str(np.exp(lnkappa)))

    annotator_acc = annotator_pseudocounts[np.arange(num_classes), np.arange(num_classes), :] \
                    / np.sum(annotator_pseudocounts, axis=1)
    annotator_acc *= (class_pseudocounts / np.sum(class_pseudocounts))[:, None]
    annotator_acc = np.sum(annotator_acc, axis=0)

    return t, annotator_acc


def ds(C, L, smooth=0.001, max_iter=10):
    '''
    The original Dawid and Skene 1979 method. This is a maximum likelihood variant of IBCC-VB.
    '''

    N = C.shape[0]
    K = C.shape[1]

    # init t using a simple Dawid and Skene implementation
    t = np.zeros((N, L)) + smooth
    for l in range(L):
        t[:, l] += np.sum(C == l, axis=1)
    t /= np.sum(t, axis=1)[:, None]

    alpha0 = smooth * np.ones((L, L, K))
    nu0 = smooth * np.ones(L)
    alpha = np.copy(alpha0)

    for i in range(max_iter):
        # print('D&S m-step')
        nu = nu0 + np.sum(t, axis=0)
        lnkappa = np.log(nu / np.sum(nu))

        for l in range(L):
            alpha[:, l, :] = alpha0[:, l, :] + t.T.dot(C == l)
        lnpi = np.log(alpha / np.sum(alpha, axis=1)[:, None, :])

        # print('D&S e-step,')

        t = lnkappa[None, :]
        for l in range(L):
            t = (C == l).dot(lnpi[:, l, :].T) + t

        t = np.exp(t)
        t /= np.sum(t, axis=1)[:, None]

    #print('Class proportions = %s' % str(np.exp(lnkappa)))

    return t