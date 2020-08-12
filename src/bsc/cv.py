import numpy as np
from scipy.special._logsumexp import logsumexp
from scipy.special.basic import psi

from bsc.annotator_model import Annotator, log_dirichlet_pdf

class VectorWorker(Annotator):
# Worker model: confusion/accuracy vector for workers ----------------------------------------------------------------------------

# the alphas are counted as for IBCC for simplicity. However, when Pi is computed, the counts for incorrect answers
# are summed together to compute lnPi_incorrect, then exp(lnPi_incorrect) is divided by nclasses - 1.

    def __init__(self, alpha0_diags, alpha0_factor, L, nModels):

        # WORKING SETUP FOR NER AND PICO:
        # for the incorrect answers, the psuedo count splits the alpha0_factor equally
        # between the incorrect answers and multiplies by 2 (why?)
        alpha0_base = alpha0_factor / (L - 1)

        # for the correct answers, the pseudo count is alpha0_factor + alpha0_diags
        alpha0_correct = alpha0_diags + alpha0_factor - alpha0_base

        self.alpha0 = alpha0_base * np.ones((L, L)) + \
                 alpha0_correct * np.eye(L)

        self.alpha0_data = {}
        self.lnPi_data = {}
        for m in range(nModels):
            self.alpha0_data[m] = np.copy(self.alpha0)
            self.alpha0_data[m][:] = alpha0_base

        self.nModels = nModels

        print(self.alpha0)


    def _init_lnPi(self):
        # init to prior
        self.alpha = np.copy(self.alpha0)
        self.alpha_data = {}
        for midx in range(self.nModels):
            self.alpha_data[midx] = np.copy(self.alpha0_data[midx])

        # Returns the initial values for alpha and lnPi
        self.lnPi = self.q_pi()

    def _calc_q_pi(self, alpha):
        '''
        Update the annotator models.
        '''
        psi_alpha_sum = psi(np.sum(alpha, 1))[:, None, :]
        self.lnpi = psi(alpha) - psi_alpha_sum

        lnpi_incorrect = psi(np.sum(alpha, 1)
                        - alpha[range(self.alpha.shape[0]), range(alpha.shape[0]), :]) \
                        - psi_alpha_sum[:, 0, :]
        lnpi_incorrect = np.log(np.exp(lnpi_incorrect) / float(alpha.shape[1] - 1)) # J x K

        for j in range(alpha.shape[0]):
            for l in range(alpha.shape[1]):
                if j == l:
                    continue
                self.lnpi[j, l, :] = lnpi_incorrect[j, :]

        return self.lnpi


    def update_post_alpha(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha.
        '''
        dims = self.alpha0.shape
        self.alpha = self.alpha0.copy()

        for j in range(dims[0]):
            Tj = E_t[:, j]

            for l in range(dims[1]):
                counts = (C == l).T.dot(Tj).reshape(-1)
                self.alpha[j, l, :] += counts


    def update_post_alpha_data(self, model_idx, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha when C is the votes for one annotator, and each column contains a probability of a vote.
        '''
        dims = self.alpha0_data.shape
        self.alpha_data[model_idx] = self.alpha0_data.copy()

        for j in range(dims[0]):
            Tj = E_t[:, j]

            for l in range(dims[1]):
                counts = (C[:, l:l+1]).T.dot(Tj).reshape(-1)
                self.alpha_data[model_idx][j, l, :] += counts


    def _read_lnPi(self, lnPi, l, C, Cprev, Krange, nscores, blanks):
        if l is None:
            if np.isscalar(Krange):
                Krange = np.array([Krange])[None, :]
            if np.isscalar(C):
                C = np.array([C])[:, None]

            result = lnPi[:, C, Krange]
            if not np.isscalar(C):
                result[:, blanks] = 0
        else:
            result = lnPi[l, C, Krange]
            if np.isscalar(C):
                if C == -1:
                    result = 0
            else:
                result[blanks] = 0

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

    def lowerbound_terms(self):
        # the dimension over which to sum, i.e. over which the values are parameters of a single Dirichlet
        sum_dim = 0 # in the case that we have only one Dirichlet per worker, e.g. accuracy model

        L = self.alpha0.shape[0]
        correctcounts = self.alpha0[range(L), range(L), :]
        incorrectcounts = np.sum(self.alpha0, axis=1) - correctcounts
        alpha0 = np.concatenate((correctcounts[:, None, :], incorrectcounts[:, None, :]), axis=1)

        correctcounts = self.alpha[range(L), range(L), :]
        incorrectcounts = np.sum(self.alpha, axis=1) - correctcounts
        alpha = np.concatenate((correctcounts[:, None, :], incorrectcounts[:, None, :]), axis=1)

        lnpicorrect = self.lnPi[range(L), range(L), :]
        lnpiincorrect = logsumexp(self.lnPi, axis=1) - lnpicorrect
        lnpi = np.concatenate((lnpicorrect[:, None, :], lnpiincorrect[:, None, :]), axis=1)

        lnpPi = log_dirichlet_pdf(alpha0, lnpi, sum_dim)

        lnqPi = log_dirichlet_pdf(alpha, lnpi, sum_dim)

        for midx, alpha0_data in enumerate(self.alpha0_data):
            correctcounts = alpha0_data[range(L), range(L)]
            incorrectcounts = np.sum(alpha0_data, axis=1) - correctcounts
            alpha0_data = np.concatenate((correctcounts, incorrectcounts), axis=1)

            correctcounts = self.alpha_data[midx][range(L), range(L)]
            incorrectcounts = np.sum(self.alpha_data[midx], axis=1) - correctcounts
            alpha_data = np.concatenate((correctcounts, incorrectcounts), axis=1)

            lnpicorrect = self.lnPi_data[midx][range(L), range(L), :]
            lnpiincorrect = np.sum(self.lnPi_data[midx], axis=1) - lnpicorrect
            lnpi_data = np.concatenate((lnpicorrect, lnpiincorrect), axis=1)

            lnpPi += log_dirichlet_pdf(alpha0_data, lnpi_data, sum_dim)
            lnqPi += log_dirichlet_pdf(alpha_data, lnpi_data, sum_dim)

        return lnpPi, lnqPi