import numpy as np
from scipy.special.basic import psi

from bayesian_combination.annotator_models.annotator_model import log_dirichlet_pdf
from bayesian_combination.annotator_models.cv import ConfusionVectorAnnotator


class ConfusionMatrixAnnotator(ConfusionVectorAnnotator):
    # Worker model: Bayesianized Dawid and Skene confusion matrix ----------------------------------------------------------

    def __init__(self, alpha0_diags, alpha0_factor, L, nModels):
        super().__init__(alpha0_diags, alpha0_factor, L, nModels)

    def init_lnPi(self, N):
        # Returns the initial values for alpha and lnPi
        psi_alpha_sum = psi(np.sum(self.alpha0, 1))
        self.lnPi = psi(self.alpha0) - psi_alpha_sum[:, None, :]

        # init to prior
        self.alpha = np.copy(self.alpha0)
        self.alpha_taggers = {}
        for midx in range(self.nModels):
            self.alpha_taggers[midx] = np.copy(self.alpha0_taggers[midx])

    def _calc_q_pi(self, alpha):
        '''
        Update the annotator models.
        '''
        psi_alpha_sum = psi(np.sum(alpha, 1))[:, None, :]
        self.lnPi = psi(self.alpha) - psi_alpha_sum
        return self.lnPi


    def update_alpha(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha.
        '''
        dims = self.alpha0.shape
        self.alpha = self.alpha0.copy()

        for l in range(dims[1]):
            self.alpha[:, l, :] += E_t.T.dot(C == l)

        # print(self.alpha[0, :, 2])


    def update_alpha_taggers(self, model_idx, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha when C is the votes for one annotator, and each column contains a probability of a vote.
        '''
        dims = self.alpha0_taggers[model_idx].shape
        self.alpha_taggers[model_idx] = self.alpha0_taggers[model_idx].copy()

        for j in range(dims[0]):
            Tj = E_t[:, j]

            for l in range(dims[1]):
                counts = (C[:, l:l+1]).T.dot(Tj).reshape(-1)
                self.alpha_taggers[model_idx][j, l, :] += counts


    def read_lnPi(self, l, C, Cprev, doc_id, Krange, nscores, blanks):

        if l is None:
            result = 0
            for l in range(self.lnPi.shape[1]):
                result += (C==l).dot(self.lnPi[:, l, :].T).T

        else:
            result = 0
            for m in range(self.lnPi.shape[1]):
                result += (C==m).dot(self.lnPi[l, m, :][:, None])
            result = result.flatten()

        return result


    def read_lnPi_taggers(self, l, C, Cprev, nscores, model_idx):

        lnPi = self.lnPi_taggers[model_idx]
        if l is None:
            result = lnPi[:, C, 0]
        else:
            result = lnPi[l, C, 0]
        return result


    def expand_alpha0(self, C, K, doc_start, nscores):
        '''
        Take the alpha0 for one worker and expand.
        :return:
        '''
        L = self.alpha0.shape[0]

        # set priors
        if self.alpha0 is None:
            # dims: true_label[t], current_annoc[t],  previous_anno c[t-1], annotator k
            self.alpha0 = np.ones((L, nscores, K)) + 1.0 * np.eye(L)[:, :, None]
        else:
            self.alpha0 = self.alpha0[:, :, None]
            self.alpha0 = np.tile(self.alpha0, (1, 1, K))

        for midx in range(self.nModels):
            if self.alpha0_taggers[midx] is None:
                self.alpha0_taggers[midx] = np.ones((L, nscores, 1)) + 1.0 * np.eye(L)[:, :, None]
            elif self.alpha0_taggers[midx].ndim == 2:
                self.alpha0_taggers[midx] = self.alpha0_taggers[midx][:, :, None]


    def _calc_EPi(self, alpha):
        return alpha / np.sum(alpha, axis=1)[:, None, :]


    def lowerbound_terms(self):
        # the dimension over which to sum, i.e. over which the values are parameters of a single Dirichlet
        sum_dim = 1 # in the case that we have multiple Dirichlets per worker, e.g. IBCC, sequential-BCC model

        lnpPi = log_dirichlet_pdf(self.alpha0, self.lnPi, sum_dim)
        lnqPi = log_dirichlet_pdf(self.alpha, self.lnPi, sum_dim)

        for midx, alpha0_data in enumerate(self.alpha0_taggers):
            lnpPi += log_dirichlet_pdf(alpha0_data, self.lnPi_taggers[midx], sum_dim)
            lnqPi += log_dirichlet_pdf(self.alpha_taggers[midx], self.lnPi_taggers[midx], sum_dim)

        return lnpPi, lnqPi