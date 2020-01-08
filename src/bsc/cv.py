import numpy as np
from scipy.special.basic import psi

from bsc.annotator_model import Annotator


class VectorWorker(Annotator):
# Worker model: confusion/accuracy vector for workers ----------------------------------------------------------------------------

# the alphas are counted as for IBCC for simplicity. However, when Pi is computed, the counts for incorrect answers
# are summed together to compute lnPi_incorrect, then exp(lnPi_incorrect) is divided by nclasses - 1.

    def __init__(self, alpha0_diags, alpha0_factor, L, nModels):

        # for the incorrect answers, the psuedo count splits the alpha0_factor equally
        # between the incorrect answers and multiplies by 2 (why?)
        alpha0_base = alpha0_factor / (L - 1)

        # for the correct answers, the pseudo count is alpha0_factor + alpha0_diags
        alpha0_correct = alpha0_diags + alpha0_factor - alpha0_base

        self.alpha0 = alpha0_base * np.ones((L, L)) + \
                 alpha0_correct * np.eye(L)

        self.alpha0_data = {}
        self.lnPi = {}
        for m in range(nModels):
            self.alpha0_data[m] = np.copy(self.alpha0)
            self.alpha0_data[m][:] = alpha0_base

        self.nModels = nModels


    def _init_lnPi(self):
        # init to prior
        self.alpha = np.copy(self.alpha0)

        # Returns the initial values for alpha and lnPi
        self.lnPi = self.q_pi()


    def _calc_q_pi(self, alpha):
        '''
        Update the annotator models.
        '''
        psi_alpha_sum = psi(np.sum(alpha, 1))[:, None, :]
        q_pi = psi(alpha) - psi_alpha_sum

        q_pi_incorrect = psi(np.sum(alpha, 1)
                        - alpha[range(self.alpha.shape[0]), range(alpha.shape[0]), :]) \
                        - psi_alpha_sum[:, 0, :]
        q_pi_incorrect = np.log(np.exp(q_pi_incorrect) / float(alpha.shape[1] - 1)) # J x K

        for j in range(alpha.shape[0]):
            for l in range(alpha.shape[1]):
                if j == l:
                    continue
                q_pi[j, l, :] = q_pi_incorrect[j, :]

        return q_pi


    def _post_alpha(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha.
        '''
        dims = self.alpha0.shape
        self.alpha = self.alpha0.copy()

        for j in range(dims[0]):
            Tj = E_t[:, j]

            for l in range(dims[1]):
                counts = (C == l + 1).T.dot(Tj).reshape(-1)
                self.alpha[j, l, :] += counts


    def _post_alpha_data(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha when C is the votes for one annotator, and each column contains a probability of a vote.
        '''
        dims = self.alpha0_data.shape
        self.alpha_data = self.alpha0_data.copy()

        for j in range(dims[0]):
            Tj = E_t[:, j]

            for l in range(dims[1]):
                counts = (C[:, l:l+1]).T.dot(Tj).reshape(-1)
                self.alpha_data[j, l, :] += counts


    def _read_lnPi(self, lnPi, l, C, Cprev, Krange, nscores, blanks=None):
        if l is None:
            if np.isscalar(Krange):
                Krange = np.array([Krange])[None, :]
            if np.isscalar(C):
                C = np.array([C])[:, None]

            result = lnPi[:, C, Krange]
            result[:, C == -1] = 0
        else:
            result = lnPi[l, C, Krange]
            if np.isscalar(C):
                if C == -1:
                    result = 0
            else:
                result[C == -1] = 0

        return result


    def _expand_alpha0(self, K, nscores):
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
            if self.alpha0_data[midx] is None:
                self.alpha0_data[midx] = np.ones((L, L, 1)) + 1.0 * np.eye(L)[:, :, None]
            elif self.alpha0_data[midx].ndim == 2:
                self.alpha0_data[midx] = self.alpha0_data[midx][:, :, None]


    def _calc_EPi(self, alpha):

        EPi = np.zeros_like(alpha)

        for j in range(alpha.shape[0]):

            EPi[j, j, :] = alpha[j, j, :] / np.sum(alpha[j, :, :], axis=0)

            EPi_incorrect_j = (np.sum(alpha[j, :, :], axis=0) - alpha[j, j, :]) \
                              / np.sum(alpha[j, :, :], axis=0)
            EPi_incorrect_j /= float(alpha.shape[1] - 1)

            for l in range(alpha.shape[1]):
                if j == l:
                    continue
                EPi[j, l, :] = EPi_incorrect_j

        return EPi