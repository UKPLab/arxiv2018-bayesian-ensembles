import numpy as np
from scipy.special.basic import psi

from bsc.annotator_model import Annotator, log_dirichlet_pdf

class AccuracyWorker(Annotator):
    # Worker model: accuracy only ------------------------------------------------------------------------------------------
    # lnPi[1] = ln p(correct)
    # lnPi[0] = ln p(wrong)

    def __init__(self, alpha0_diags, alpha0_factor, L, nModels):

        # for the incorrect answers, the pseudo count is alpha0_factor
        self.alpha0 = alpha0_factor * np.ones((2))

        # for the correct answers, the pseudo count is alpha0_factor + alpha0_diags
        self.alpha0[1] += alpha0_diags  # diags are bias toward correct answer

        self.alpha0_data = {}
        self.lnPi = {}
        for m in range(nModels):
            self.alpha0_data[m] = np.copy(self.alpha0)
            self.alpha0_data[m][:] = alpha0_factor

        self.nModels = nModels


    def _init_lnPi(self):

        # Returns the initial values for alpha and lnPi
        psi_alpha_sum = psi(np.sum(self.alpha0, 0))
        self.lnPi = psi(self.alpha0) - psi_alpha_sum[None, :]

        # init to prior
        self.alpha = np.copy(self.alpha0)
        self.alpha_data = {}
        for midx in range(self.nModels):
            self.alpha_data[midx] = np.copy(self.alpha0_data[midx])


    def _calc_q_pi(self, alpha):
        '''
        Update the annotator models.
        '''
        psi_alpha_sum = psi(np.sum(alpha, 0))[None, :]
        self.lnPi = psi(alpha) - psi_alpha_sum

        return self.lnPi


    def update_post_alpha(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha.
        '''
        nclasses = E_t.shape[1]
        self.alpha = self.alpha0.copy()

        for j in range(nclasses):
            Tj = E_t[:, j]

            correct_count = (C == j).T.dot(Tj).reshape(-1)
            self.alpha[1, :] += correct_count

            for l in range(nscores):
                if l == j:
                    continue
                incorrect_count = (C == l).T.dot(Tj).reshape(-1)
                self.alpha[0, :] += incorrect_count


    def update_post_alpha_data(self, model_idx, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha when C is the votes for one annotator, and each column contains a probability of a vote.
        '''
        nclasses = E_t.shape[1]
        self.alpha_data[model_idx] = self.alpha0_data[model_idx].copy()

        for j in range(nclasses):
            Tj = E_t[:, j]

            correct_count = C[:, j:j+1].T.dot(Tj).reshape(-1)
            self.alpha_data[model_idx][1, :] += correct_count

            for l in range(nscores):
                if l == j:
                    continue
                incorrect_count = C[:, l:l+1].T.dot(Tj).reshape(-1)
                self.alpha_data[model_idx][0, :] += incorrect_count


    def _read_lnPi(self, lnPi, l, C, Cprev, Krange, nscores, blanks):

        if np.isscalar(C):
            N = 1
        else:
            N = C.shape[0]

        if np.isscalar(Krange):
            K = 1
        else:
            K = Krange.shape[-1]

        if l is None:
            result = np.zeros((nscores, N, K))
            for l in range(nscores):

                if np.isscalar(C) and l == C:
                    result[l, :, :] = lnPi[1, Krange]

                elif np.isscalar(C) and l != C:
                    result[l, :, :] = lnPi[0, Krange]
                    # incorrect answers: split mass across classes evenly
                    result[l, :, :] = np.log(np.exp(result[l, :, :]) / float(nscores))

                else:
                    idx = (l==C).astype(int)
                    result[l, :] = lnPi[idx, Krange]
                    # incorrect answers: split mass across classes evenly
                    result[l, idx == 0] = np.log(np.exp(result[l, idx == 0]) / float(nscores))
                    result[l, blanks] = 0

            return result

        if np.isscalar(C) and l == C:
            result = lnPi[1, Krange]

        elif np.isscalar(C) and l != C:
            result = lnPi[0, Krange]
            # incorrect answers: split mass across classes evenly
            result = np.log(np.exp(result) / float(nscores))

        else:
            idx = (l==C).astype(int)
            result = lnPi[idx, Krange]
            # incorrect answers: split mass across classes evenly
            result[idx == 0] = np.log(np.exp(result[idx==0]) / float(nscores))

            result[blanks] = 0

        return result


    def _expand_alpha0(self, K, nscores):
        '''
        Take the alpha0 for one worker and expand.
        :return:
        '''
        # set priors
        if self.alpha0 is None:
            # dims: true_label[t], current_annoc[t],  previous_anno c[t-1], annotator k
            self.alpha0 = np.ones((2, K))
            self.alpha0[1, :] += 1.0
        else:
            self.alpha0 = self.alpha0[:, None]
            self.alpha0 = np.tile(self.alpha0, (1, K))

        for midx in range(self.nModels):
            if self.alpha0_data[midx] is None:
                self.alpha0_data[midx] = np.ones((2, 1))
                self.alpha0_data[midx][1, :] += 1.0
            elif self.alpha0_data[midx].ndim == 1:
                self.alpha0_data[midx] = self.alpha0_data[midx][:, None]


    def _calc_EPi(self, alpha):
        return alpha / np.sum(alpha, axis=0)[None, :]


    def lowerbound_terms(self):
        # the dimension over which to sum, i.e. over which the values are parameters of a single Dirichlet
        sum_dim = 0 # in the case that we have only one Dirichlet per worker, e.g. accuracy model

        lnpPi = log_dirichlet_pdf(self.alpha0, self.lnPi, sum_dim)
        lnqPi = log_dirichlet_pdf(self.alpha, self.lnPi, sum_dim)

        for midx, alpha0_data in enumerate(self.alpha0_data):
            lnpPi += log_dirichlet_pdf(alpha0_data, self.lnPi_data[midx], sum_dim)
            lnqPi += log_dirichlet_pdf(self.alpha_data[midx], self.lnPi_data[midx], sum_dim)

        return lnpPi, lnqPi