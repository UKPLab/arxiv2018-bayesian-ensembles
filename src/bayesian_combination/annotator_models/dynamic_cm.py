import numpy as np
from scipy.special.basic import psi

from bayesian_combination.annotator_models.cm import ConfusionMatrixAnnotator


def state_to_alpha(logodds, var):
    '''
    Map the latent state moments to parameters of a Beta distribution.
    :param logodds:
    :param var:
    :return: the first beta parameter, alpha, and the sum of the beta parameters alpha+beta
    '''
    alpha1 = 1/var * (1+np.exp(logodds))
    alpha2 = alpha1 * (1+np.exp(-logodds))
    return alpha1, alpha2

def alpha_to_state(alpha, l):
    '''
    Map alpha parameters of a Dirichlet to mean and variance parameters of hidden state.
    :param alpha:
    :param l:
    :return: mean vector and 2-D covariance vector.
    '''
    counts = alpha[:, l]
    other_counts = np.sum(alpha, axis=1) - alpha[:, l]

    # if np.any(counts < 1e-300):
    #     print('alpha_to_state fix small counts')
    #     counts[counts < 1e-300] = 1e-300 # artificial bottom to prevent NaNs
    # if np.any(other_counts < 1e-300):
    #     print('alpha_to_state fix small other_counts')
    #     other_counts[other_counts < 1e-300] = 1e-300 # artificial bottom to prevent NaNs

    Wmean = np.log(counts) - np.log(other_counts)
    Wmean = Wmean.flatten()
    diag_vals = (1 / counts) + (1 / (other_counts))
    P = np.diagflat(diag_vals)
    return Wmean, P


class DynamicConfusionMatrixAnnotator(ConfusionMatrixAnnotator):
    # A variant of the method used in DynIBCC, which tracks changes to annotator behaviour over time. ------------------
    # We do apply the dynamic confusion matrix only to the annotators, not the taggers. The integrated taggers make
    # predictions at one time using the same model, so their behaviour does not vary over time.

    def __init__(self, alpha0_diags, alpha0_factor, L, nModels, prior_type='consistent'):
        super().__init__(alpha0_diags, alpha0_factor, L, nModels)
        self.L = L

        self.prior_type = prior_type # Are the priors applied to the start of the sequence only, or to whole sequence?
        # Values may be 'initial' or 'consistent'.


    def expand_alpha0(self, C, _, doc_start, nscores):
        '''
        Take the alpha0 for one worker and expand.
        :return:
        '''
        L = self.alpha0.shape[0]

        label_idxs = np.argwhere(C[doc_start.flatten(), :].T != -1)
        self.annotator_idxs = label_idxs[:, 0]
        self.doc_ids = label_idxs[:, 1] # index of the document that each confusion matrix corresponds to
        self.doc_ids_by_token = np.cumsum(doc_start)-1 # index of the document that each token belongs to
        self.Nannos = len(self.annotator_idxs)

        label_idxs = np.argwhere(C.T != -1)
        self.annotator_by_tok_ids = label_idxs[:, 0]
        self.tok_ids = label_idxs[:, 1]
        self.Ntokannos = len(self.tok_ids)

        self.confmat_ids = np.zeros(self.Ntokannos, dtype=int)
        for n in range(self.Ntokannos):
            self.confmat_ids[n] = np.argwhere((self.annotator_by_tok_ids[n] == self.annotator_idxs) &
                                                (self.doc_ids_by_token[self.tok_ids[n]] == self.doc_ids))[0][0]

        self.u_annotators = np.unique(self.annotator_idxs)
        self.K = len(self.u_annotators)

        # set priors
        if self.alpha0 is None:
            # dims: true_label[t], current_annoc[t],  previous_anno c[t-1], annotator k
            self.alpha0 = np.ones((L, nscores, self.Nannos)) + 1.0 * np.eye(L)[:, :, None]
        else:
            self.alpha0 = self.alpha0[:, :, None]
            self.alpha0 = np.tile(self.alpha0, (1, 1, self.Nannos))

        for midx in range(self.nModels):
            if self.alpha0_taggers[midx] is None:
                self.alpha0_taggers[midx] = np.ones((L, nscores, 1)) + 1.0 * np.eye(L)[:, :, None]
            elif self.alpha0_taggers[midx].ndim == 2:
                self.alpha0_taggers[midx] = self.alpha0_taggers[midx][:, :, None]


    def _post_Alpha_binary(self, C, E_t, l):
        # l is the observation index into alpha

        # FILTERING -- UPDATES GIVEN PREVIOUS TIMESTEPS
        # p(\pi_t | data up to and including t)

        # Variables used in smoothing step but calculated during filtering
        Wmean_po = np.zeros((self.L, self.Ntokannos))

        P_po = {}  # np.zeros((self.nclasses * self.K, self.nclasses * self.K, self.N))
        Kalman = {}  # np.zeros((self.nclasses * self.K, self.K, self.N))
        # indicates which blocks of the kalman matrix are used

        # filtering variables
        q = np.zeros(self.Ntokannos)
        I = np.eye(self.L)
        eta_pr = np.zeros((self.Ntokannos))
        r_pr = np.zeros((self.Ntokannos))
        u_pr = np.ones(C.shape[1])
        # loop through each row of the table in turn (each row is a timestep). Tau also corresponds to the object ID
        n = 0  # the data point index
        while n < self.Ntokannos:
            if np.mod(n, 100000)==99999:
                print("Alpha update filter for label %i: %i / %i" % (l, n, self.Nannos))

            h_cov = E_t[self.tok_ids[n], :][:, None]

            # initialise the state priors
            if n==0 or self.annotator_by_tok_ids[n] != self.annotator_by_tok_ids[n-1]:
                # prior is the prior over first state only
                if self.prior_type == 'initial':
                    confmat_id = self.confmat_ids[n]
                    Wmean_pr, P_pr = alpha_to_state(self.alpha0[:, :, confmat_id], l)
                else:
                    Wmean_pr, P_pr = alpha_to_state(np.ones((self.L, self.L)), 0)
            else:
                Wmean_pr = Wmean_po[:, n - 1]
                P_pr = P_po[n - 1]

                if self.confmat_ids[n] != self.confmat_ids[n-1]:
                    P_pr += np.eye(self.L) * q[n-1]

            # priors in latent state form
            eta_pr[n] = h_cov.T.dot(Wmean_pr)
            r_pr[n] = h_cov.T.dot(P_pr).dot(h_cov)

            # map hidden state priors to prior alphas
            alpha_tilde_pr, alpha_tilde_pr_sum = state_to_alpha(eta_pr[n], r_pr[n])

            # get the current observation
            c = np.sum(C[self.tok_ids[n], self.annotator_by_tok_ids[n]] == l)

            # update to get posterior given current time-step
            alpha_tilde_po = alpha_tilde_pr + c
            alpha_tilde_po_sum = alpha_tilde_pr_sum + 1

            # state change variance used in next iteration: if the posterior is more uncertain, we estimate it by
            # taking the different in variance. If the posterior is less or equally uncertain, we assume no state change
            if self.confmat_ids[n] != self.confmat_ids[n-1]:
                pi_tilde_pr = alpha_tilde_pr / alpha_tilde_pr_sum # prior expected pi
                u_pr[self.annotator_by_tok_ids[n]] = pi_tilde_pr * (1 - pi_tilde_pr) # prior variance in pi

            pi_tilde_po = alpha_tilde_po / alpha_tilde_po_sum  # posterior expected pi
            u_po = pi_tilde_po * (1 - pi_tilde_po)  # posterior variance in pi

            q[n] = (u_po > u_pr[self.annotator_by_tok_ids[n]]) * (u_po - u_pr[self.annotator_by_tok_ids[n]])

            # get posteriors in latent state form
            other_counts = alpha_tilde_po_sum - alpha_tilde_po
            if other_counts > alpha_tilde_po/1e-300:
                print('update_alpha fixing small other counts')
                other_counts = 1e-300

            eta_po = np.log(alpha_tilde_po / other_counts)
            r_po = (1.0 / alpha_tilde_po) + (1.0 / other_counts)

            # Kalman update equations
            Kalman[n] = (P_pr.T.dot(h_cov) / r_pr[n])  # J
            R = 1 - (r_po / r_pr[n])

            # update from this observation at tau
            z = eta_po - eta_pr[n]

            Wmean_po[:, n] = Wmean_pr + Kalman[n].T * z
            P_po[n] = P_pr - (Kalman[n].dot(h_cov.T).dot(P_pr) * R)

            # increment the timestep counter
            n += 1

        # SMOOTHING -- UPDATES GIVEN ALL TIMESTEPS
        # pi(\pi_t | all data up to time self.N)
        while n > 0:
            n -= 1
            if np.mod(n, 100000)==99999:
                print("Alpha update smoother for label %i: %i/%i" % (l, n, self.Nannos))

            if n == self.Ntokannos-1 or self.annotator_by_tok_ids[n] != self.annotator_by_tok_ids[n+1]:
                lambda_mean = np.zeros(self.L)
                Lambda_cov = np.zeros((self.L, self.L))

            h_cov = E_t[self.tok_ids[n], :][:, None]

            delta_Wmean = P_po[n].dot(lambda_mean)
            delta_P = P_po[n].dot(Lambda_cov).dot(P_po[n].T)

            Wmean_po[:, n] = Wmean_po[:, n] - delta_Wmean
            P_po[n] = P_po[n] - delta_P

            eta_po = h_cov.T.dot(Wmean_po[:, n])
            r_po = h_cov.T.dot(P_po[n]).dot(h_cov)
            z = eta_po - eta_pr[n]
            z_rpr = z / r_pr[n]

            R = 1 - (r_po / r_pr[n])
            B = I - Kalman[n].dot(h_cov.T)

            lambda_mean = B.T.dot(lambda_mean) - h_cov.flatten() * z_rpr
            Lambda_cov = B.T.dot(Lambda_cov).dot(B) + (h_cov * R / r_pr[n]).dot(h_cov.T)

            pi_var = np.diag(P_po[n])
            alpha_l, alphasum = state_to_alpha(Wmean_po[:, n], pi_var)

            confmat_id = self.confmat_ids[n]

            self.alpha[:, l, confmat_id] = alpha_l
            if self.L == 2 and l==1:
                self.alpha[:, 0, confmat_id] = alphasum - self.alpha[:, 1, confmat_id]


    def update_alpha(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        self.alpha = np.empty(self.alpha0.shape)

        if self.L>2:
            for l in range(self.L):
                self._post_Alpha_binary(C, E_t, l)
        else:
            self._post_Alpha_binary(C, E_t, 1)

        if self.prior_type == 'consistent':
            self.alpha += self.alpha0


    def read_lnPi(self, l, C, Cprev, doc_ids, Krange, nscores, blanks):
        label_idxs = np.argwhere(C != -1)
        tok_idxs = label_idxs[:, 0]
        annotator_idxs = label_idxs[:, 1]

        if doc_ids is None:
            doc_ids = self.doc_ids_by_token

        # map doc id and annotator id to confusion matrix
        confmat_ids = (annotator_idxs[:, None] == self.annotator_idxs[None, :]) & (doc_ids[tok_idxs, None] == self.doc_ids[None, :])
        confmat_idxs = np.argwhere(confmat_ids)
        confmat_ids = confmat_idxs[:, 1]

        if l is None:
            if not np.any(C != -1):
                return np.zeros((self.L, C.shape[0]))

            result = np.zeros((self.L, C.shape[0], C.shape[1]))
            for m in range(self.lnPi.shape[1]):
                result_m = (C[C!=-1]==m)[None, :] * self.lnPi[:, m, confmat_ids]
                result[:, tok_idxs, annotator_idxs] += result_m
        else:
            if not np.any(C != -1):
                return np.zeros(C.shape[0])

            result = np.zeros(C.shape)
            for m in range(self.lnPi.shape[1]):
                result_m = (C[C!=-1]==m) * self.lnPi[l, m, confmat_ids]
                result[tok_idxs, annotator_idxs] += result_m

        result = np.sum(result, axis=-1)

        return result


    def _calc_EPi(self, alpha):
        EPi = alpha / np.sum(alpha, axis=1)[:, None, :]

        meanEPi = np.zeros((EPi.shape[0], EPi.shape[1], self.K))
        for aidx, a in enumerate(self.u_annotators):
            meanEPi = np.mean(EPi[:, :, self.annotator_idxs==a], axis=2)

        return meanEPi

