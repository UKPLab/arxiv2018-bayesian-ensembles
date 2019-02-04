import numpy as np
from scipy.special.basic import psi

from bsc.cv import VectorWorker


class ConfusionMatrixWorker(VectorWorker):
    # Worker model: Bayesianized Dawid and Skene confusion matrix ----------------------------------------------------------

    def _init_lnPi(alpha0):
        # Returns the initial values for alpha and lnPi
        psi_alpha_sum = psi(np.sum(alpha0, 1))
        lnPi = psi(alpha0) - psi_alpha_sum[:, None, :]

        # init to prior
        alpha = np.copy(alpha0)
        return alpha, lnPi

    def _calc_q_pi(alpha):
        '''
        Update the annotator models.
        '''
        psi_alpha_sum = psi(np.sum(alpha, 1))[:, None, :]
        q_pi = psi(alpha) - psi_alpha_sum
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
                counts = (C == l + 1).T.dot(Tj)
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
            alpha0_data = np.ones((L, nscores, 1)) + 1.0 * np.eye(L)[:, :, None]
        elif alpha0_data.ndim == 2:
            alpha0_data = alpha0_data[:, :, None]

        return alpha0, alpha0_data

    def _calc_EPi(alpha):
        return alpha / np.sum(alpha, axis=1)[:, None, :]