import numpy as np
from scipy.special import psi

def ds(C, L, smooth=0.001, max_iter=10):

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

    print('Class proportions = %s' % str(np.exp(lnkappa)))

    return t

def ibccvb(C, L, nu0=0.001, alpha0=0.1, alpha_diag=0.9, max_iter=10):

    N = C.shape[0]
    K = C.shape[1]

    # init t using a simple Dawid and Skene implementation
    t = np.zeros((N, L)) + nu0
    for l in range(L):
        t[:, l] += np.sum(C == l, axis=1)
    t /= np.sum(t, axis=1)[:, None]

    alpha0 = alpha0 * np.ones((L, L, K)) + np.eye(L)[:, :, None] * alpha_diag
    nu0 = nu0 * np.ones(L)
    alpha = np.copy(alpha0)

    for i in range(max_iter):
        # print('D&S m-step')
        nu = nu0 + np.sum(t, axis=0)
        lnkappa = psi(nu) - psi(np.sum(nu))

        for l in range(L):
            alpha[:, l, :] = alpha0[:, l, :] + t.T.dot(C == l)
        lnpi = psi(alpha) - psi(np.sum(alpha, axis=1)[:, None, :])

        # print('D&S e-step,')

        t = lnkappa[None, :]
        for l in range(L):
            t = (C == l).dot(lnpi[:, l, :].T) + t

        t = np.exp(t)
        t /= np.sum(t, axis=1)[:, None]

    print('Class proportions = %s' % str(np.exp(lnkappa)))

    return t