import numpy as np
from scipy.special.basic import psi


class VectorWorker():
# Worker model: confusion/accuracy vector for workers ----------------------------------------------------------------------------

# the alphas are counted as for IBCC for simplicity. However, when Pi is computed, the counts for incorrect answers
# are summed together to compute lnPi_incorrect, then exp(lnPi_incorrect) is divided by nclasses - 1.

    def _init_alpha0(alpha0_diags, alpha0_factor, L):
        alpha0_factor = alpha0_factor / ((L - 1) / 2)
        alpha0_diags = alpha0_diags + alpha0_factor - alpha0_factor

        alpha0 = alpha0_factor * np.ones((L, L)) + \
                          alpha0_diags * np.eye(L)

        alpha0_data = np.copy(self.bac_alpha0)
        alpha0_data[:] = alpha0_factor / ((L - 1) / 2) + np.eye(L) * (
                alpha0_diags + alpha0_factor - (alpha0_factor / ((L - 1) / 2)))

        return alpha0, alpha0_data

    def _init_lnPi(alpha0):
        # Returns the initial values for alpha and lnPi
        lnPi = VectorWorker._calc_q_pi(alpha0)

        # init to prior
        alpha = np.copy(alpha0)
        return alpha, lnPi

    def _calc_q_pi(alpha):
        '''
        Update the annotator models.

        TODO Representing using a full matrix might break lower bound implementation
        '''
        psi_alpha_sum = psi(np.sum(alpha, 1))[:, None, :]
        q_pi = psi(alpha) - psi_alpha_sum

        q_pi_incorrect = psi(np.sum(alpha, 1) - alpha[np.arange(alpha.shape[0]), np.arange(alpha.shape[0]), :]) \
                         - psi_alpha_sum[:, 0, :]
        q_pi_incorrect = np.log(np.exp(q_pi_incorrect) / float(alpha.shape[1] - 1)) # J x K

        for j in range(alpha.shape[0]):
            for l in range(alpha.shape[1]):
                if j == l:
                    continue
                q_pi[j, l, :] = q_pi_incorrect[j, :]

        return q_pi

    def _post_alpha(E_t, C, alpha0, alpha, doc_start, nscores, before_doc_idx=-1):  # Posterior Hyperparameters
        '''
        Update alpha.
        '''
        dims = alpha0.shape
        alpha = alpha0.copy()

        for j in range(dims[0]):
            Tj = E_t[:, j]

            for l in range(dims[1]):
                counts = (C == l + 1).T.dot(Tj).reshape(-1)
                alpha[j, l, :] += counts

        return alpha

    def _post_alpha_data(E_t, C, alpha0, alpha, doc_start, nscores, before_doc_idx=-1):  # Posterior Hyperparameters
        '''
        Update alpha when C is the votes for one annotator, and each column contains a probability of a vote.
        '''
        dims = alpha0.shape
        alpha = alpha0.copy()

        for j in range(dims[0]):
            Tj = E_t[:, j]

            for l in range(dims[1]):
                counts = (C[:, l:l+1]).T.dot(Tj).reshape(-1)
                alpha[j, l, :] += counts

        return alpha

    def _read_lnPi(lnPi, l, C, Cprev, Krange, nscores):
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

    def _expand_alpha0(alpha0, alpha0_data, K, nscores):
        '''
        Take the alpha0 for one worker and expand.
        :return:
        '''
        L = alpha0.shape[0]

        # set priors
        if alpha0 is None:
            # dims: true_label[t], current_annoc[t],  previous_anno c[t-1], annotator k
            alpha0 = np.ones((L, nscores, K)) + 1.0 * np.eye(L)[:, :, None]
        else:
            alpha0 = alpha0[:, :, None]
            alpha0 = np.tile(alpha0, (1, 1, K))

        if alpha0_data is None:
            alpha0_data = np.ones((L, L, 1)) + 1.0 * np.eye(L)[:, :, None]
        elif alpha0_data.ndim == 2:
            alpha0_data = alpha0_data[:, :, None]

        return alpha0, alpha0_data

    def _calc_EPi(alpha):

        EPi = np.zeros_like(alpha)

        for j in range(alpha.shape[0]):

            EPi[j, j, :] = alpha[j, j, :] / np.sum(alpha[j, :, :], axis=0)

            EPi_incorrect_j = (np.sum(alpha[j, :, :], axis=0) - alpha[j, j, :]) / np.sum(alpha[j, :, :], axis=0)
            EPi_incorrect_j /= float(alpha.shape[1] - 1)

            for l in range(alpha.shape[1]):
                if j == l:
                    continue
                EPi[j, l, :] = EPi_incorrect_j

        return EPi