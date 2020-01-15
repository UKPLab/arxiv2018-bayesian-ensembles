import numpy as np
from scipy.special import logsumexp
from scipy.special.basic import psi

from bsc.annotator_model import Annotator, log_dirichlet_pdf


class MACEWorker(Annotator):
    # Worker model: MACE-like spammer model --------------------------------------------------------------------------------

    # alpha[0,:] and alpha[1,:] are parameters for the spamming probability
    # alpha[2:2+nscores,:] are parameters for the spamming pattern
    # similarly for lnPi:
    # lnPi[1, :] = ln p(correct answer)
    # lnPi[0, :] = ln p(incorrect/spam answer)
    # lnPi[2:2+nscores, :] = ln p(label given worker is spamming/incorrect)

    def __init__(self, alpha0_diags, alpha0_factor, L, nModels):
        # for incorrect answers and for spamming pattern, pseudocount is alpha0_factor
        self.alpha0 = alpha0_factor * np.ones((2 + L))
        # for corrent answers, pseudocount is alpha0_factor + alpha0_diags
        self.alpha0[1] = alpha0_diags  # diags are bias toward correct answer

        self.alpha0_data = {}
        self.lnPi = {}
        for m in range(nModels):
            self.alpha0_data[m] = np.copy(self.alpha0)
            self.alpha0_data[m][:] = alpha0_factor

        self.nModels = nModels


    def _init_lnPi(self):
        # Returns the initial values for alpha and lnPi

        psi_alpha_sum = np.zeros_like(self.alpha0)
        psi_alpha_sum[0, :] = psi(self.alpha0[0,:] + self.alpha0[1, :])
        psi_alpha_sum[1, :] = psi_alpha_sum[0, :]

        psi_alpha_sum[2:, :] = psi(np.sum(self.alpha0[2:, :], 0))[None, :]

        self.lnPi = psi(self.alpha0) - psi_alpha_sum

        # init to prior
        self.alpha = np.copy(self.alpha0)
        self.alpha_data = {}
        for midx in range(self.nModels):
            self.alpha_data[midx] = np.copy(self.alpha0_data[midx])


    def _calc_q_pi(self, alpha):
        '''
        Update the annotator models.
        '''
        psi_alpha_sum = np.zeros_like(alpha)
        psi_alpha_sum[0, :] = psi(alpha[0,:] + alpha[1, :])
        psi_alpha_sum[1, :] = psi_alpha_sum[0, :]
        psi_alpha_sum[2:, :] = psi(np.sum(alpha[2:, :], 0))[None, :]

        return psi(alpha) - psi_alpha_sum


    def update_post_alpha(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha.
        '''
        # Reusing some equations from the Java MACE implementation.,,
        # strategyMarginal[i,k] = <scalar per worker> p(k knows vs k is spamming for item i | pi, C, E_t)? = ...
        # ... = \sum_j{ E_t[i, j] / (pi[0,k]*pi[2+C[i,k],k] + pi[1,k]*[C[i,k]==j] } * pi[0,k] * pi[2+C[i,k],k]
        # instanceMarginal = \sum_j p(t_i = j) = term used for normalisation
        # spamming = competence = accuracy = pi[1]
        # a = annotator
        # d = item number
        # ai = index of annotator a's annotation for item d
        # goldlabelmarginals[d] = p(C, t_i = j) =   prior(t_i=j) * \prod_k (pi[0,k] * pi[2+C[i,k],k] + pi[1,k] * [C[i,k]==j])
        # [labels[d][ai]] = C[i, :]
        # thetas = pi[2:,:] = strategy params
        # strategyExpectedCounts[a][labels[d][ai]] = pseudo-count for each spamming action = alpha[2+C[i,k], k] += ...
        # ... += strategyMarginal[i,k] / instanceMarginal
        # knowingExpectedCounts[a][0]+=strategyMarginal/instanceMarginal ->alpha[0,k]+=strategyMarginal/instanceMarginal
        # knowingExpectedCounts[a][1] += (goldLabelMarginals[d][labels[d][ai]] * spamming[a][1] / (spamming[a][0] *
        # ...thetas[a][labels[d][ai]] + spamming[a][1])) / instanceMarginal;
        # ... -> alpha[1,k] += E_t[i, C[i,k]] * pi[1,k] / (pi[0,k]*pi[2+C[i,k],k] + pi[1,k]) / instanceMarginal
        # ... everything is normalised by instanceMarginal because goldlabelMarginals is not normalised and is actually
        # a joint probability

        # start by determining the probability of not spamming at each data point using current estimates of pi
        pknowing = 0
        pspamming = 0

        Pi = np.zeros_like(self.alpha)
        Pi[0, :] = self.alpha[0, :] / (self.alpha[0, :] + self.alpha[1, :])
        Pi[1, :] = self.alpha[1, :] / (self.alpha[0, :] + self.alpha[1, :])
        Pi[2:, :] = self.alpha[2:, :] / np.sum(self.alpha[2:, :], 0)[None, :]

        pspamming_j_unnormed = Pi[0, :][None, :] * Pi[C + 2, np.arange(C.shape[1])[None, :]]

        for j in range(E_t.shape[1]):
            Tj = E_t[:, j:j+1]

            pknowing_j_unnormed = (Pi[1,:][None, :] * (C == j))

            pknowing_j = pknowing_j_unnormed / (pknowing_j_unnormed + pspamming_j_unnormed)
            pspamming_j = pspamming_j_unnormed / (pknowing_j_unnormed + pspamming_j_unnormed)

            # The cases where worker has not given a label are not really spam!
            pspamming_j[C==-1] = 0

            pknowing += pknowing_j * Tj
            pspamming += pspamming_j * Tj

        correct_count = np.sum(pknowing, 0)
        incorrect_count = np.sum(pspamming, 0)

        self.alpha[1, :] = self.alpha0[1, :] + correct_count
        self.alpha[0, :] = self.alpha0[0, :] + incorrect_count

        for l in range(nscores):
            strategy_count_l = np.sum((C == l) * pspamming, 0)
            self.alpha[l+2, :] = self.alpha0[l+2, :] + strategy_count_l


    def update_post_alpha_data(self, model_idx, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha when C is the votes for one annotator, and each column contains a probability of a vote.
        '''

        # start by determining the probability of not spamming at each data point using current estimates of pi
        pknowing = 0
        pspamming = 0

        Pi = np.zeros_like(self.alpha)
        Pi[0, :] = self.alpha_data[model_idx][0, :] / (self.alpha_data[model_idx][0, :] + self.alpha_data[model_idx][1, :])
        Pi[1, :] = self.alpha_data[model_idx][1, :] / (self.alpha_data[model_idx][0, :] + self.alpha_data[model_idx][1, :])
        Pi[2:, :] = self.alpha_data[model_idx][2:, :] / np.sum(self.alpha_data[model_idx][2:, :], 0)[None, :]

        pspamming_j_unnormed = 0
        for j in range(C.shape[1]):
            pspamming_j_unnormed += Pi[0, :] * Pi[j, :] * C[:, j:j+1]

        for j in range(E_t.shape[1]):
            Tj = E_t[:, j:j+1]

            pknowing_j_unnormed = (Pi[1,:][None, :] * (C[:, j:j+1]))

            pknowing_j = pknowing_j_unnormed / (pknowing_j_unnormed + pspamming_j_unnormed)
            pspamming_j = pspamming_j_unnormed / (pknowing_j_unnormed + pspamming_j_unnormed)

            pknowing += pknowing_j * Tj
            pspamming += pspamming_j * Tj

        correct_count = np.sum(pknowing, 0)
        incorrect_count = np.sum(pspamming, 0)

        self.alpha_data[model_idx][1, :] = self.alpha0_data[model_idx][1, :] + correct_count
        self.alpha_data[model_idx][0, :] = self.alpha0_data[model_idx][0, :] + incorrect_count

        for l in range(nscores):
            strategy_count_l = np.sum((C[:, l:l+1]) * pspamming, 0)
            self.alpha_data[model_idx][l+2, :] = self.alpha0_data[model_idx][l+2, :] + strategy_count_l


    def _read_lnPi(self, lnPi, l, C, Cprev, Krange, nscores, blanks):

        ll_incorrect = lnPi[0, Krange] + lnPi[C+2, Krange]

        if np.isscalar(C):
            N = 1
            if C == -1:
                ll_incorrect = 0
        else:
            N = C.shape[0]
            ll_incorrect[C == -1] = 0

        if np.isscalar(Krange):
            K = 1
        else:
            K = Krange.shape[-1]

        if l is None:
            ll_correct = np.zeros((nscores, N, K))
            for m in range(nscores):

                if np.isscalar(C) and C == m:
                    ll_correct[m] = lnPi[1, Krange]

                elif np.isscalar(C) and C != m:
                    ll_correct[m] = - np.inf

                else:
                    idx = (C == m).astype(int)

                    ll_correct[m] = lnPi[1, Krange] * idx
                    ll_correct[m, idx==0] = -np.inf

            ll_incorrect = np.tile(ll_incorrect, (nscores, 1, 1))
        else:
            if np.isscalar(C) and C == l:
                ll_correct = lnPi[1, Krange]

            elif np.isscalar(C) and C != l:
                ll_correct = - np.inf

            else:
                idx = (C == l).astype(int)
                ll_correct = lnPi[1, Krange] * idx
                ll_correct[idx == 0] = - np.inf

        result = logsumexp([ll_correct, ll_incorrect], axis=0)
        if not np.isscalar(C) and l is None:
            result[:, blanks] = 0
        elif not np.isscalar(C):
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
            self.alpha0 = np.ones((nscores + 2, K))
            self.alpha0[1, :] += 1.0
        else:
            self.alpha0 = self.alpha0[:, None]
            self.alpha0 = np.tile(self.alpha0, (1, K))

        for midx in range(self.nModels):
            if self.alpha0_data[midx] is None:
                self.alpha0_data[midx] = np.ones((nscores + 2, 1))
                self.alpha0_data[midx][1, :] += 1.0
            elif self.alpha0_data[midx].ndim == 1:
                self.alpha0_data[midx] = self.alpha0_data[midx][:, None]


    def _calc_EPi(self, alpha):
        pi = np.zeros_like(alpha)

        pi[0] = alpha[0] / (alpha[0] + alpha[1])
        pi[1] = alpha[1] / (alpha[0] + alpha[1])
        pi[2:] = alpha[2:] / np.sum(alpha[2:], axis=0)[None, :]

        return pi

    def lowerbound_terms(self):
        # the dimension over which to sum, i.e. over which the values are parameters of a single Dirichlet
        sum_dim = 0 # in the case that we have multiple Dirichlets per worker, e.g. IBCC, sequential-BCC model

        lnpPi_correct = log_dirichlet_pdf(self.alpha0[0:2, :], self.lnPi[0:2, :], sum_dim)
        lnpPi_strategy = log_dirichlet_pdf(self.alpha0[2:, :], self.lnPi[2:, :], sum_dim)

        lnqPi_correct = log_dirichlet_pdf(self.alpha[0:2, :], self.lnPi[0:2, :], sum_dim)
        lnqPi_strategy = log_dirichlet_pdf(self.alpha[2:, :], self.lnPi[2:, :], sum_dim)

        for midx in range(len(self.alpha0_data)):
            lnpPi_correct += log_dirichlet_pdf(self.alpha0_data[midx][0:2, :], self.lnPi_data[midx][0:2, :], sum_dim)
            lnpPi_strategy += log_dirichlet_pdf(self.alpha0_data[midx][2:, :], self.lnPi_data[midx][2:, :], sum_dim)

            lnqPi_correct += log_dirichlet_pdf(self.alpha_data[midx][0:2, :], self.lnPi_data[midx][0:2, :], sum_dim)
            lnqPi_strategy += log_dirichlet_pdf(self.alpha_data[midx][2:, :], self.lnPi_data[midx][2:, :], sum_dim)

        return lnpPi_correct + lnpPi_strategy, lnqPi_correct + lnqPi_strategy