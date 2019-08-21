import numpy as np
from scipy.special import logsumexp
from scipy.special.basic import psi


class MACEWorker():
    # Worker model: MACE-like spammer model --------------------------------------------------------------------------------

    # alpha[0,:] and alpha[1,:] are parameters for the spamming probability
    # alpha[2:2+nscores,:] are parameters for the spamming pattern
    # similarly for lnPi:
    # lnPi[1, :] = ln p(correct answer)
    # lnPi[0, :] = ln p(incorrect/spam answer)
    # lnPi[2:2+nscores, :] = ln p(label given worker is spamming/incorrect)

    def _init_alpha0(alpha0_diags, alpha0_factor, L):
        alpha0 = alpha0_factor * np.ones((2 + L))
        alpha0[1] = alpha0_diags  # diags are bias toward correct answer

        alpha0_data = np.copy(alpha0)
        alpha0_data[:] = alpha0_factor
        alpha0_data[1] = alpha0_diags

        return alpha0, alpha0_data

    def _init_lnPi(alpha0):
        # Returns the initial values for alpha and lnPi

        psi_alpha_sum = np.zeros_like(alpha0)
        psi_alpha_sum[0, :] = psi(alpha0[0,:] + alpha0[1, :])
        psi_alpha_sum[1, :] = psi_alpha_sum[0, :]

        psi_alpha_sum[2:, :] = psi(np.sum(alpha0[2:, :], 0))[None, :]

        lnPi = psi(alpha0) - psi_alpha_sum

        # init to prior
        alpha = np.copy(alpha0)

        return alpha, lnPi

    def _calc_q_pi(alpha):
        '''
        Update the annotator models.
        '''
        psi_alpha_sum = np.zeros_like(alpha)
        psi_alpha_sum[0, :] = psi(alpha[0,:] + alpha[1, :])
        psi_alpha_sum[1, :] = psi_alpha_sum[0, :]
        psi_alpha_sum[2:, :] = psi(np.sum(alpha[2:, :], 0))[None, :]

        ElnPi = psi(alpha) - psi_alpha_sum

        # ElnPi[0, :] = np.log(0.5)
        # ElnPi[1, :] = np.log(0.5)
        # ElnPi[2:, :] = np.log(1.0 / float(alpha.shape[1] - 2))

        return ElnPi

    def _post_alpha(E_t, C, alpha0, alpha, doc_start, nscores, before_doc_idx=-1):  # Posterior Hyperparameters
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

        Pi = np.zeros_like(alpha)
        Pi[0, :] = alpha[0, :] / (alpha[0, :] + alpha[1, :])
        Pi[1, :] = alpha[1, :] / (alpha[0, :] + alpha[1, :])
        Pi[2:, :] = alpha[2:, :] / np.sum(alpha[2:, :], 0)[None, :]

        pspamming_j_unnormed = Pi[0, :][None, :] * Pi[C + 1, np.arange(C.shape[1])[None, :]]

        for j in range(E_t.shape[1]):
            Tj = E_t[:, j:j+1]

            pknowing_j_unnormed = (Pi[1,:][None, :] * (C == j + 1))

            pknowing_j = pknowing_j_unnormed / (pknowing_j_unnormed + pspamming_j_unnormed)
            pspamming_j = pspamming_j_unnormed / (pknowing_j_unnormed + pspamming_j_unnormed)

            # The cases where worker has not given a label are not really spam!
            pspamming_j[C==0] = 0

            pknowing += pknowing_j * Tj
            pspamming += pspamming_j * Tj

        correct_count = np.sum(pknowing, 0)
        incorrect_count = np.sum(pspamming, 0)

        alpha[1, :] = alpha0[1, :] + correct_count
        alpha[0, :] = alpha0[0, :] + incorrect_count

        for l in range(nscores):
            strategy_count_l = np.sum((C == l + 1) * pspamming, 0)
            alpha[l+2, :] = alpha0[l+2, :] + strategy_count_l

        return alpha

    def _post_alpha_data(E_t, C, alpha0, alpha, doc_start, nscores, before_doc_idx=-1):  # Posterior Hyperparameters
        '''
        Update alpha when C is the votes for one annotator, and each column contains a probability of a vote.
        '''

        # start by determining the probability of not spamming at each data point using current estimates of pi
        pknowing = 0
        pspamming = 0

        Pi = np.zeros_like(alpha)
        Pi[0, :] = alpha[0, :] / (alpha[0, :] + alpha[1, :])
        Pi[1, :] = alpha[1, :] / (alpha[0, :] + alpha[1, :])
        Pi[2:, :] = alpha[2:, :] / np.sum(alpha[2:, :], 0)[None, :]

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

        alpha[1, :] = alpha0[1, :] + correct_count
        alpha[0, :] = alpha0[0, :] + incorrect_count

        for l in range(nscores):
            strategy_count_l = np.sum((C[:, l:l+1]) * pspamming, 0)
            alpha[l+2, :] = alpha0[l+2, :] + strategy_count_l

        return alpha

    def _read_lnPi(lnPi, l, C, Cprev, Krange, nscores):

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

        return logsumexp([ll_correct, ll_incorrect], axis=0)

    def _expand_alpha0(alpha0, alpha0_data, K, nscores, uniform_priors):
        '''
        Take the alpha0 for one worker and expand.
        :return:
        '''
        L = alpha0.shape[0]

        # set priors
        if alpha0 is None:
            # dims: true_label[t], current_annoc[t],  previous_anno c[t-1], annotator k
            alpha0 = np.ones((nscores + 2, K))
            alpha0[1, :] += 1.0
        else:
            alpha0 = alpha0[:, None]
            alpha0 = np.tile(alpha0, (1, K))

        alpha0[:, uniform_priors] = alpha0[0, uniform_priors]

        if alpha0_data is None:
            alpha0_data = np.ones((nscores + 2, 1))
            alpha0_data[1, :] += 1.0
        elif alpha0_data.ndim == 1:
            alpha0_data = alpha0_data[:, None]


        return alpha0, alpha0_data

    def _calc_EPi(alpha):

        pi = np.zeros_like(alpha)

        pi[0] = alpha[0] / (alpha[0] + alpha[1])
        pi[1] = alpha[1] / (alpha[0] + alpha[1])
        pi[2:] = alpha[2:] / np.sum(alpha[2:], axis=0)[None, :]

        return pi