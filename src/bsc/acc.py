import numpy as np
from scipy.special.basic import psi


class AccuracyWorker():
    # Worker model: accuracy only ------------------------------------------------------------------------------------------
    # lnPi[1] = ln p(correct)
    # lnPi[0] = ln p(wrong)

    def _init_alpha0(alpha0_diags, alpha0_factor, L):
        alpha0 = alpha0_factor * np.ones((2))
        alpha0[1] += alpha0_diags  # diags are bias toward correct answer

        alpha0_data = np.copy(alpha0)
        alpha0_data[:] = alpha0_factor
        alpha0_data[1] += alpha0_diags

        return alpha0, alpha0_data

    def _init_lnPi(alpha0):

        # Returns the initial values for alpha and lnPi
        psi_alpha_sum = psi(np.sum(alpha0, 0))
        lnPi = psi(alpha0) - psi_alpha_sum[None, :]

        # init to prior
        alpha = np.copy(alpha0)
        return alpha, lnPi

    def _calc_q_pi(alpha):
        '''
        Update the annotator models.
        '''
        psi_alpha_sum = psi(np.sum(alpha, 0))[None, :]
        q_pi = psi(alpha) - psi_alpha_sum
        return q_pi

    def _post_alpha(E_t, C, alpha0, alpha, doc_start, nscores, before_doc_idx=-1):  # Posterior Hyperparameters
        '''
        Update alpha.
        '''
        nclasses = E_t.shape[1]
        alpha = alpha0.copy()

        for j in range(nclasses):
            Tj = E_t[:, j]

            correct_count = (C == j + 1).T.dot(Tj).reshape(-1)
            alpha[1, :] += correct_count

            for l in range(nscores):
                if l == j:
                    continue
                incorrect_count = (C == l + 1).T.dot(Tj).reshape(-1)
                alpha[0, :] += incorrect_count

        return alpha

    def _post_alpha_data(E_t, C, alpha0, alpha, doc_start, nscores, before_doc_idx=-1):  # Posterior Hyperparameters
        '''
        Update alpha when C is the votes for one annotator, and each column contains a probability of a vote.
        '''
        nclasses = E_t.shape[1]
        alpha = alpha0.copy()

        for j in range(nclasses):
            Tj = E_t[:, j]

            correct_count = C[:, j:j+1].T.dot(Tj).reshape(-1)
            alpha[1, :] += correct_count

            for l in range(nscores):
                if l == j:
                    continue
                incorrect_count = C[:, l:l+1].T.dot(Tj).reshape(-1)
                alpha[0, :] += incorrect_count

        return alpha

    def _read_lnPi(lnPi, l, C, Cprev, Krange, nscores):

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
                    result[l, :, :] = lnPi[idx, Krange]
                    # incorrect answers: split mass across classes evenly
                    result[l, idx == 0] = np.log(np.exp(result[l, idx == 0]) / float(nscores))

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

        return result

    def _expand_alpha0(alpha0, alpha0_data, K, nscores):
        '''
        Take the alpha0 for one worker and expand.
        :return:
        '''

        # set priors
        if alpha0 is None:
            # dims: true_label[t], current_annoc[t],  previous_anno c[t-1], annotator k
            alpha0 = np.ones((2, K))
            alpha0[1, :] += 1.0
        else:
            alpha0 = alpha0[:, None]
            alpha0 = np.tile(alpha0, (1, K))

        if alpha0_data is None:
            alpha0_data = np.ones((2, 1))
            alpha0_data[1, :] += 1.0
        elif alpha0_data.ndim == 1:
            alpha0_data = alpha0_data[:, None]

        return alpha0, alpha0_data

    def _calc_EPi(alpha):
        return alpha / np.sum(alpha, axis=0)[None, :]