import numpy as np

from bayesian_combination.annotator_models.annotator_model import log_dirichlet_pdf
from scipy.special import psi, logsumexp


class LabelModel(object):
    '''
    Wrapper around the GP tagger to use it as a label model. The difference is that the BC model allows only one
    label model, which is fully trusted; multiple taggers can be integrated and each tagger has a confusion matrix that
    accounts for its reliability.
    '''

    def __init__(self, beta0, L, verbose=False):
        self.L = L
        self.verbose = verbose

        self.beta0 = beta0
        self.beta_shape = beta0.shape

    def set_beta0(self, beta0):
        self.beta0 = beta0

    def update_B(self, features, ln_features_likelihood, predict_only=False):
        pass

    def init_t(self, C, doc_start, A, blanks, gold=None):

        self.C = C
        self.ds = doc_start.flatten() == 1
        self.A = A
        self.N = C.shape[0]
        self.blanks = blanks
        self.gold = gold

        self.beta = self.beta0

        if self.beta0.ndim >= 2:
            beta0_sum = psi(np.sum(self.beta0, -1))[:, None]
        else:
            beta0_sum = psi(np.sum(self.beta0))

        self.lnB = psi(self.beta0) - beta0_sum
        self.lnB = np.tile(self.lnB, (self.N, 1))

        self.Cprev = np.append(np.zeros((1, self.C.shape[1]), dtype=int) - 1, self.C[:-1, :], axis=0)

    def update_t(self, parallel, C_data):
        pass

    def most_likely_labels(self, word_ll):
        pass

    def lowerbound_terms(self):
        pass


class IndependentLabelModel(LabelModel):

    def update_B(self, features, ln_features_likelihood, predict_only=False):
        '''
        Update the transition model.

        q_t_joint is N x L x L. In this case, we just sum up over all previous label values, so the counts are
        independent of the previous labels.
        '''
        self.beta = self.beta0 + np.sum(self.Et, 0)

        self.lnB = psi(self.beta) - psi(np.sum(self.beta, -1))

        self.lnB = self.lnB[None, :] + ln_features_likelihood

        if np.any(np.isnan(self.lnB)):
            print('_update_B: nan value encountered!')


    def init_t(self, C, doc_start, A, blanks, gold=None):

        super().init_t(C, doc_start, A, blanks, gold)

        # initialise using the fraction of votes for each label
        self.Et = np.zeros((C.shape[0], self.L)) + self.beta0
        for l in range(self.L):
            self.Et[:, l] += np.sum(C == l, axis=1)
        self.Et /= np.sum(self.Et, axis=1)[:, None]

        return self.Et


    def update_t(self, parallel, C_data):
        Krange = np.arange(self.C.shape[1], dtype=int)

        self.lnp_Ct = np.copy(self.lnB)

        lnPi_data_terms = 0
        for model_idx in range(len(C_data)):

            C_m = C_data[model_idx]
            C_m_prev = np.concatenate((np.zeros((1, self.L)), C_data[model_idx][:-1, :]), axis=0)

            for m in range(self.L):
                for n in range(self.L):
                    lnPi_data_terms += self.A.read_lnPi_taggers(None, m, n, self.L, model_idx)[:, None] * \
                                       C_m[:, m][None, :] * C_m_prev[:, n][None, :]

        Elnpi = self.A.read_lnPi(None, self.C, self.Cprev, None, Krange[None, :], self.L, self.blanks)
        self.lnp_Ct += (Elnpi + lnPi_data_terms).T

        p_Ct = np.exp(self.lnp_Ct)
        self.Et = p_Ct / np.sum(p_Ct, axis=1)[:, None]

        if self.gold is not None:
            goldidxs = self.gold != -1
            self.Et[goldidxs, :] = 0
            self.Et[goldidxs, self.gold[goldidxs]] = 1.0

        return self.Et

    def most_likely_labels(self, word_ll):
        return np.max(self.Et, axis=1), np.argmax(self.Et, axis=1)


    def lowerbound_terms(self):
        lnB = psi(self.beta) - psi(np.sum(self.beta, -1))
        lnB[np.isinf(lnB) | np.isnan(lnB)] = 0

        lnpt = self.lnp_Ct * self.Et
        lnpt = np.sum(lnpt)

        lnqt = np.log(self.Et)
        lnqt[self.Et == 0] = 0
        lnqt *= self.Et
        lnqt = np.sum(lnqt)

        lnpB = log_dirichlet_pdf(self.beta0, lnB, 0)
        lnqB = log_dirichlet_pdf(self.beta, lnB, 0)

        return lnpt + lnpB, lnqt + lnqB

# ----------------------------------------------------------------------------------------------------------------------

def _doc_forward_pass(doc_id, C, C_data, blanks, lnB, worker_model, outside_label):
    '''
    Perform the forward pass of the Forward-Backward bayesian_combination (for a single document).
    '''
    T = C.shape[0]  # infer number of tokens
    L = lnB.shape[-1]  # infer number of labels
    K = C.shape[-1]  # infer number of annotators
    Krange = np.arange(K)

    # initialise variables
    lnR_ = np.zeros((T, L))
    scaling = np.zeros(T)
    # blanks = C == -1
    # For the first data point
    lnPi_terms = worker_model.read_lnPi(None, C[0:1, :], np.zeros((1, K), dtype=int)-1, np.zeros(1) + doc_id,
                                        Krange[None, :], L, blanks[0:1])
    lnPi_data_terms = 0

    if C_data is not None:
        for m in range(L):
            for model_idx in range(len(C_data)):
                lnPi_data_terms += (worker_model.read_lnPi_taggers(None, m, outside_label, L, model_idx)[:, None]
                                    * C_data[model_idx][0, m])[:, 0]

    lnR_[0, :] = lnB[0, outside_label, :] + lnPi_terms[:, 0] + lnPi_data_terms
    scaling[0] = logsumexp(lnR_[0, :])
    lnR_[0, :] = lnR_[0, :] - scaling[0]

    Cprev = C[:-1, :]
    Ccurr = C[1:, :]

    # the other data points
    lnPi_terms = worker_model.read_lnPi(None, Ccurr, Cprev, doc_id + np.zeros(Ccurr.shape[0]), Krange[None, :], L, blanks[1:])
    lnPi_data_terms = 0
    if C_data is not None:
        for m in range(L):
            for n in range(L):
                for model_idx in range(len(C_data)):
                    lnPi_data_terms += worker_model.read_lnPi_taggers(None, m, n, L, model_idx)[:, None] * \
                                       C_data[model_idx][1:, m][None, :] * C_data[model_idx][:-1, n][None, :]

    likelihood_next = lnPi_terms + lnPi_data_terms  # L x T-1
    # lnPi[:, Ccurr, Cprev, Krange[None, :]]

    # iterate through all tokens, starting at the beginning going forward
    sum_trans_likelihood = lnB[1:] + likelihood_next.T[:, None, :]

    for t in range(1, T):
        # iterate through all possible labels
        # prev_idx = Cprev[t - 1, :]
        lnR_t = logsumexp(lnR_[t - 1, :][:, None] + sum_trans_likelihood[t-1], axis=0)
        # , np.sum(mask[t, :] * lnPi[:, C[t, :] - 1, prev_idx, Krange], axis=1)

        # normalise
        scaling[t] = logsumexp(lnR_t)
        lnR_[t, :] = lnR_t - scaling[t]

    return lnR_, scaling


def _doc_backward_pass(doc_id, C, C_data, blanks, lnB, L, worker_model, scaling):
    '''
    Perform the backward pass of the Forward-Backward bayesian_combination (for a single document).
    '''
    # infer number of tokens, labels, and annotators
    T = C.shape[0]
    K = C.shape[-1]
    Krange = np.arange(K)

    # initialise variables
    lnLambda = np.zeros((T, L))

    Ccurr = C[:-1, :]
    Cnext = C[1:, :]
    blanks = blanks[1:]#Cnext == -1

    lnPi_terms = worker_model.read_lnPi(None, Cnext, Ccurr, np.zeros(Cnext.shape[0]) + doc_id, Krange[None, :], L, blanks)
    lnPi_data_terms = 0
    if C_data is not None:
        for model_idx in range(len(C_data)):

            C_data_curr = C_data[model_idx][:-1, :]
            C_data_next = C_data[model_idx][1:, :]

            for m in range(L):
                for n in range(L):
                    terms_mn = worker_model.read_lnPi_taggers(None, m, n, L, model_idx)[:, None] \
                               * C_data_next[:, m][None, :] \
                               * C_data_curr[:, n][None, :]
                    terms_mn[:, (C_data_next[:, m] * C_data_curr[:, n]) == 0] = 0
                    lnPi_data_terms += terms_mn

    likelihood_next = lnPi_terms + lnPi_data_terms

    # iterate through all tokens, starting at the end going backwards
    for t in range(T - 2, -1, -1):
        # logsumexp over the L classes of the next timestep
        lnLambda_t = logsumexp(lnLambda[t + 1, :][None, :] + lnB[t+1, :, :] + likelihood_next[:, t][None, :], axis=1)

        # logsumexp over the L classes of the current timestep to normalise
        lnLambda[t] = lnLambda_t - scaling[t+1]  # logsumexp(lnLambda_t)

    if np.any(np.isnan(lnLambda)):
        print('backward pass: nan value encountered at indexes: ')
        print(np.argwhere(np.isnan(lnLambda)))

    return lnLambda


def _doc_most_probable_sequence(doc_id, C, C_data, lnEB, L, K, worker_model, before_doc_idx):
    lnV = np.zeros((C.shape[0], L))
    prev = np.zeros((C.shape[0], L), dtype=int)  # most likely previous states

    for l in range(L):
        if l != before_doc_idx:
            lnEB[0, l, :] = -np.inf

    blanks = C == -1
    Cprev = np.concatenate((np.zeros((1, C.shape[1]), dtype=int) - 1, C[:-1, :]), axis=0)

    ds = np.zeros((C.shape[0], 1)).astype(bool)
    ds[0] = True

    lnPi_terms = worker_model.read_lnPi(None, C, Cprev, np.zeros(C.shape[0])+doc_id, np.arange(K), L, blanks)
    lnPi_data_terms = 0
    if C_data is not None:
        for model_idx, C_data_m in enumerate(C_data):
            C_data_m_prev_0 = np.zeros((1, L), dtype=int)
            C_data_m_prev_0[0, before_doc_idx] = 1
            C_data_m_prev = np.concatenate((C_data_m_prev_0, C_data_m[:-1, :]), axis=0)
            for m in range(L):
                for n in range(L):
                    weights = (C_data_m[:, m] * C_data_m_prev[:, n])[None, :]
                    terms_mn = weights * worker_model.read_lnPi_taggers(None, m, n, L, model_idx)[:, None]
                    terms_mn[:, weights[0] == 0] = 0

                    lnPi_data_terms += terms_mn

    likelihood = lnPi_terms + lnPi_data_terms

    for t in range(0, C.shape[0]):
        for l in range(L):
            p_current = lnV[t - 1, :] + lnEB[t, :, l] + likelihood[l, t]
            lnV[t, l] = np.max(p_current)
            prev[t, l] = np.argmax(p_current, axis=0)

        lnV[t, :] -= logsumexp(lnV[t, :])  # normalise to prevent underflow in long sequences

    # decode
    seq = np.zeros(C.shape[0], dtype=int)

    t = C.shape[0] - 1

    seq[t] = np.argmax(lnV[t, :])
    pseq = np.exp(np.max(lnV[t, :]))

    for t in range(C.shape[0] - 2, -1, -1):
        seq[t] = prev[t + 1, seq[t + 1]]

    return pseq, seq