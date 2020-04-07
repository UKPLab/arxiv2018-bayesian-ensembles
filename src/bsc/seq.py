import numpy as np
from scipy.special.basic import psi

from bsc.annotator_model import log_dirichlet_pdf
from bsc.cv import VectorWorker


class SequentialWorker(VectorWorker):
    # Worker model: sequential model of workers-----------------------------------------------------------------------------

    def __init__(self, alpha0_diags, alpha0_factor, L, nModels, tagging_scheme,
                 beginning_labels, inside_labels, outside_label, alpha0_B_factor, rare_transition_pseudocount):
        super().__init__(alpha0_diags, alpha0_factor, L, nModels)

        self.tagging_scheme = tagging_scheme
        self.beginning_labels = beginning_labels
        self.inside_labels = inside_labels
        self.outside_label = outside_label
        self.alpha0_B_factor = alpha0_B_factor
        self.rare_transition_pseudocount = rare_transition_pseudocount
        self.C_binary = {}

    def _init_lnPi(self):
        # Returns the initial values for alpha and lnPi
        psi_alpha_sum = psi(np.sum(self.alpha0, 1))
        self.lnPi = psi(self.alpha0) - psi_alpha_sum[:, None, :, :]
        self.lnPi_data = {}

        # init to prior
        self.alpha = np.copy(self.alpha0)
        self.alpha_data = {}
        for midx in range(self.nModels):
            self.alpha_data[midx] = np.copy(self.alpha0_data[midx])


    def _calc_q_pi(self, alpha):
        '''
        Update the annotator models.
        '''
        psi_alpha_sum = psi(np.sum(alpha, 1))[:, None, :, :]
        self.lnPi = psi(alpha) - psi_alpha_sum
        return self.lnPi


    def update_post_alpha(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha.
        '''
        if len(self.C_binary) == 0:
            self.C_binary[-1] = C == -1
            for l in range(nscores):
                self.C_binary[l] = C == l

        self.alpha[:] = self.alpha0

        not_doc_starts = np.invert(doc_start[1:])

        for l in range(nscores):
            # counts of C==l for doc starts
            counts = (self.C_binary[l] * doc_start).T.dot(E_t)

            # get counts where C == l
            C_eq_l_counts = self.C_binary[l][1:, :] * not_doc_starts

            # add counts of where previous tokens are missing.
            counts +=  (C_eq_l_counts * self.C_binary[-1][:-1, :]).T.dot(E_t[1:])
            self.alpha[:, l, self.outside_label, :] += counts.T

            # add counts for all other preceding tokens
            for m in range(nscores):
                counts = (C_eq_l_counts * self.C_binary[m][:-1, :]).T.dot(E_t[1:])
                self.alpha[:, l, m, :] += counts.T


    def update_post_alpha_data(self, model_idx, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha when C is the votes for one annotator, and each column contains a probability of a vote.
        '''
        if model_idx not in self.alpha_data:
            self.alpha_data[model_idx] = self.alpha0_data[model_idx].copy()
        else:
            self.alpha_data[model_idx][:] = self.alpha0_data[model_idx]

        for j in range(nscores):
            Tj = E_t[:, j]

            for l in range(nscores):

                counts = ((C[:,l:l+1]) * doc_start).T.dot(Tj).reshape(-1)
                self.alpha_data[model_idx][j, l, self.outside_label, :] += counts

                for m in range(nscores):
                    counts = (C[:, l:l+1][1:, :] * (1 - doc_start[1:]) * C[:, m:m+1][:-1, :]).T.dot(Tj[1:]).reshape(-1)
                    self.alpha_data[model_idx][j, l, m, :] += counts


    def _read_lnPi(self, lnPi, l, C, Cprev, Krange, nscores, blanks):
        # shouldn't it use the outside factor instead of Cprev if there is a doc start?

        if l is None:
            if np.isscalar(Krange):
                Krange = np.array([Krange])[None, :]
            if np.isscalar(C):
                C = np.array([C])[:, None]
                if Cprev == -1:
                    Cprev = self.outside_label
            else:
                blanks_prev = Cprev == -1
                if np.any(blanks_prev):
                    Cprev = np.copy(Cprev)
                    Cprev[Cprev == -1] = self.outside_label

            result = lnPi[:, C, Cprev, Krange]

            if not np.isscalar(C):
                result[:, blanks] = 0
        else:
            if np.isscalar(C):
                if C == -1:
                    result = 0
                else:
                    if Cprev == -1:
                        Cprev = self.outside_label
                    result = lnPi[l, C, Cprev, Krange]
            else:
                if not np.isscalar(Cprev):
                    blanks_prev = Cprev == -1
                    if np.any(blanks_prev):
                        Cprev = np.copy(Cprev)
                        Cprev[Cprev == -1] = self.outside_label

                result = lnPi[l, C, Cprev, Krange]
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
            self.alpha0 = np.ones((L, nscores, nscores, K)) + 1.0 * np.eye(L)[:, :, None, None]
        else:
            self.alpha0 = self.alpha0[:, :, None, None]
            self.alpha0 = np.tile(self.alpha0, (1, 1, nscores, K))

        for midx in range(self.nModels):
            if self.alpha0_data[midx] is None:
                self.alpha0_data[midx] = np.ones((L, L, L, 1)) + 1.0 * np.eye(L)[:, :, None, None]
            elif self.alpha0_data[midx].ndim == 2:
                self.alpha0_data[midx] = self.alpha0_data[midx][:, :, None, None]
                self.alpha0_data[midx] = np.tile(self.alpha0_data[midx], (1, 1, L, 1))

        if self.tagging_scheme == 'IOB':
            restricted_labels = self.beginning_labels  # labels that cannot follow an outside label
            unrestricted_labels = self.inside_labels  # labels that can follow any label
        elif self.tagging_scheme == 'IOB2':
            restricted_labels = self.inside_labels
            unrestricted_labels = self.beginning_labels

        # The outside labels often need a stronger bias because they are way more frequent
        # we ought to change this to test whether amping up the outside labels really helps!
        self.alpha0[self.beginning_labels, self.beginning_labels] *= self.alpha0_B_factor
        # self.alpha0[self.outside_label, self.outside_label] *= self.alpha0_B_factor

        # set priors for pseudo-counts invalid transitions (to low values)
        for i, restricted_label in enumerate(restricted_labels):
            # Not allowed: from outside to inside
            disallowed_count = self.alpha0[:, restricted_label, self.outside_label] - self.rare_transition_pseudocount

            self.alpha0[:, unrestricted_labels[i], self.outside_label] += disallowed_count # previous experiments use this
            self.alpha0[:, restricted_label, self.outside_label] = self.rare_transition_pseudocount

            for midx in range(self.nModels):
                disallowed_count = self.alpha0_data[midx][:, restricted_label, self.outside_label] - self.rare_transition_pseudocount
                self.alpha0_data[midx][:, unrestricted_labels[i], self.outside_label] += disallowed_count
                self.alpha0_data[midx][:, restricted_label, self.outside_label] = self.rare_transition_pseudocount

            # Ban jumps from a B of one type to an I of another type
            for typeid, other_unrestricted_label in enumerate(unrestricted_labels):
                # prevent transitions from unrestricted to restricted if they don't have the same types
                if typeid == i:  # same type is allowed
                    continue

                # *** These seem to have a positive effect
                disallowed_count = self.alpha0[:, restricted_label, other_unrestricted_label] - self.rare_transition_pseudocount
                # # pseudocount is (alpha0 - 1) but alpha0 can be < 1. Removing the pseudocount maintains the relative weights between label values
                self.alpha0[:, restricted_labels[typeid], other_unrestricted_label] += disallowed_count
                # set the disallowed transition to as close to zero as possible
                self.alpha0[:, restricted_label, other_unrestricted_label] = self.rare_transition_pseudocount

                for midx in range(self.nModels):
                    disallowed_count = self.alpha0_data[midx][:, restricted_label, other_unrestricted_label, :] - self.rare_transition_pseudocount
                    self.alpha0_data[midx][:, restricted_labels[typeid], other_unrestricted_label] += disallowed_count # sticks to wrong type
                    self.alpha0_data[midx][:, restricted_label, other_unrestricted_label] = self.rare_transition_pseudocount

            # Ban jumps between Is of different types
            for typeid, other_restricted_label in enumerate(restricted_labels):
                if typeid == i:
                    continue

                # set the disallowed transition to as close to zero as possible
                # disallowed_count = self.alpha0[:, restricted_label, other_restricted_label, :] - self.rare_transition_pseudocount
                # self.alpha0[:, other_restricted_label, other_restricted_label, :] += disallowed_count #TMP
                self.alpha0[:, restricted_label, other_restricted_label] = self.rare_transition_pseudocount

                for midx in range(self.nModels):
                    # disallowed_count = self.alpha0_data[midx][:, restricted_label, other_restricted_label, :] - self.rare_transition_pseudocount
                    # self.alpha0_data[midx][:, other_restricted_label, other_restricted_label, :] += disallowed_count  # TMP sticks to wrong type
                    self.alpha0_data[midx][:, restricted_label, other_restricted_label] = self.rare_transition_pseudocount

    def _calc_EPi(self, alpha):
        return alpha / np.sum(alpha, axis=1)[:, None, :, :]


    def lowerbound_terms(self):
        # the dimension over which to sum, i.e. over which the values are parameters of a single Dirichlet
        sum_dim = 1 # in the case that we have multiple Dirichlets per worker, e.g. IBCC, sequential-BCC model

        lnpPi = log_dirichlet_pdf(self.alpha0, self.lnPi, sum_dim)
        lnqPi = log_dirichlet_pdf(self.alpha, self.lnPi, sum_dim)

        for midx, alpha0_data in enumerate(self.alpha0_data):
            lnpPi += log_dirichlet_pdf(alpha0_data, self.lnPi_data[midx], sum_dim)
            lnqPi += log_dirichlet_pdf(self.alpha_data[midx], self.lnPi_data[midx], sum_dim)

        return lnpPi, lnqPi