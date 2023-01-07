import numpy as np
from scipy.special._logsumexp import logsumexp
from scipy.special.basic import psi

from bayesian_combination.annotator_models.annotator_model import Annotator, log_dirichlet_pdf

class ConfusionVectorAnnotator(Annotator):
# Worker model: confusion/accuracy vector for workers ----------------------------------------------------------------------------

# the alphas are counted as for IBCC for simplicity. However, when Pi is computed, the counts for incorrect answers
# are summed together to compute lnPi_incorrect, then exp(lnPi_incorrect) is divided by nclasses - 1.

    def __init__(self, alpha0_diags, alpha0_factor, L, nModels):

        # for the incorrect answers, the psuedo count splits the alpha0_factor equally
        # between the incorrect answers
        alpha0_base = alpha0_factor / (L - 1)

        # for the correct answers, the pseudo count is alpha0_factor + alpha0_diags
        alpha0_correct = alpha0_diags + alpha0_factor - alpha0_base

        self.alpha0 = alpha0_base * np.ones((L, L)) + \
                 alpha0_correct * np.eye(L)

        self.alpha0_taggers = {}
        self.lnPi_taggers = {}
        for m in range(nModels):
            self.alpha0_taggers[m] = np.copy(self.alpha0)
            self.alpha0_taggers[m][:] = alpha0_base

        self.nModels = nModels


    def init_lnPi(self, N):
        # init to prior
        self.alpha = np.copy(self.alpha0)
        self.alpha_taggers = {}
        for midx in range(self.nModels):
            self.alpha_taggers[midx] = np.copy(self.alpha0_taggers[midx])

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

    def update_alpha(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
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
            if np.isscalar(Krange):
                Krange = np.array([Krange])[None, :]

            result = self.lnPi[:, C, Krange]
            if not np.isscalar(C):
                result[:, blanks] = 0
        else:
            result = self.lnPi[l, C, Krange]
            result[blanks] = 0

        return np.sum(result, axis=-1)

    def read_lnPi_taggers(self, l, C, Cprev, nscores, model_idx):
        if l is None:
            return self.lnPi_taggers[model_idx][:, C, 0]
        else:
            return self.lnPi_taggers[model_idx][l, C, 0]

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
                self.alpha0_taggers[midx] = np.ones((L, L, 1)) + 1.0 * np.eye(L)[:, :, None]
            elif self.alpha0_taggers[midx].ndim == 2:
                self.alpha0_taggers[midx] = self.alpha0_taggers[midx][:, :, None]

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

        for midx, alpha0_taggers in enumerate(self.alpha0_taggers):
            correctcounts = alpha0_taggers[range(L), range(L)]
            incorrectcounts = np.sum(alpha0_taggers, axis=1) - correctcounts
            alpha0_taggers = np.concatenate((correctcounts, incorrectcounts), axis=1)

            correctcounts = self.alpha_taggers[midx][range(L), range(L)]
            incorrectcounts = np.sum(self.alpha_taggers[midx], axis=1) - correctcounts
            alpha_taggers = np.concatenate((correctcounts, incorrectcounts), axis=1)

            lnpicorrect = self.lnPi_taggers[midx][range(L), range(L), :]
            lnpiincorrect = np.sum(self.lnPi_taggers[midx], axis=1) - lnpicorrect
            lnpi_taggers = np.concatenate((lnpicorrect, lnpiincorrect), axis=1)

            lnpPi += log_dirichlet_pdf(alpha0_taggers, lnpi_taggers, sum_dim)
            lnqPi += log_dirichlet_pdf(alpha_taggers, lnpi_taggers, sum_dim)

        return lnpPi, lnqPi
