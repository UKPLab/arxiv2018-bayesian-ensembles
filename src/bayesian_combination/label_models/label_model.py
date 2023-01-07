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
