import numpy as np
from scipy.special.basic import psi

from bayesian_combination.annotator_models.annotator_model import log_dirichlet_pdf
from bayesian_combination.annotator_models.cv import ConfusionVectorAnnotator


class SequentialAnnotator(ConfusionVectorAnnotator):
    # Worker model: sequential model of workers-------------------------------------------------------------------------

    def __init__(self, alpha0_diags, alpha0_factor, L, nModels, tagging_scheme,
                 beginning_labels, inside_labels, outside_label, alpha0_B_factor):
        super().__init__(alpha0_diags, alpha0_factor, L, nModels)

        if outside_label == L:
            self.num_prev_labels = L + 1
        else:
            self.num_prev_labels = L

        self.tagging_scheme = tagging_scheme
        self.beginning_labels = beginning_labels
        self.inside_labels = inside_labels
        self.outside_label = outside_label
        self.alpha0_B_factor = alpha0_B_factor
        self.rare_transition_pseudocount = 1e-12
        self.C_lm = {}

    def init_lnPi(self, N):
        # Returns the initial values for alpha and lnPi
        psi_alpha_sum = psi(np.sum(self.alpha0, 1))
        self.lnPi = psi(self.alpha0) - psi_alpha_sum[:, None, :, :]
        self.lnPi_taggers = {}

        # init to prior
        self.alpha = np.copy(self.alpha0)
        self.alpha_taggers = {}
        for midx in range(self.nModels):
            self.alpha_taggers[midx] = np.copy(self.alpha0_taggers[midx])

    def _calc_q_pi(self, alpha):
        """
        Update the annotator models.
        """
        psi_alpha_sum = psi(np.sum(alpha, 1))[:, None, :, :]
        self.lnPi = psi(alpha) - psi_alpha_sum
        return self.lnPi

    def update_alpha(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        """
        Update alpha.
        """
        # cache all the counts that we only need to compute once
        if len(self.C_lm) == 0:

            not_doc_starts = np.invert(doc_start[1:])

            self.C_eq_l_ds_counts = {}
            self.C_eq_l_restart_counts = {}

            C_binary = {}
            C_binary[-1] = C == -1

            for l in range(nscores):
                C_binary[l] = C == l

            for l in range(nscores):
                self.C_lm[l] = {}
                self.C_eq_l_ds_counts[l] = (C_binary[l] * doc_start).T
                C_eq_l_counts = C_binary[l][1:, :] * not_doc_starts
                self.C_eq_l_restart_counts[l] = (C_eq_l_counts * C_binary[-1][:-1, :]).T
                for m in range(nscores):
                    self.C_lm[l][m] = (C_eq_l_counts * C_binary[m][:-1, :]).T

        self.alpha[:] = self.alpha0

        for l in range(nscores):
            # counts of C==l for doc starts
            counts = self.C_eq_l_ds_counts[l].dot(E_t)

            # add counts of where previous tokens are missing.
            if np.any(self.C_eq_l_restart_counts[l]):
                counts += self.C_eq_l_restart_counts[l].dot(E_t[1:])

            self.alpha[:, l, self.outside_label, :] += counts.T

            # add counts for all other preceding tokens
            for m in range(nscores):
                counts = self.C_lm[l][m].dot(E_t[1:])
                self.alpha[:, l, m, :] += counts.T

    def update_alpha_taggers(self, model_idx, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        """
        Update alpha when C is the votes for one annotator, and each column contains a probability of a vote.
        """
        if model_idx not in self.alpha_taggers:
            self.alpha_taggers[model_idx] = self.alpha0_taggers[model_idx].copy()
        else:
            self.alpha_taggers[model_idx][:] = self.alpha0_taggers[model_idx]

        for j in range(nscores):
            Tj = E_t[:, j]

            for l in range(nscores):

                counts = ((C[:,l:l+1]) * doc_start).T.dot(Tj).reshape(-1)
                self.alpha_taggers[model_idx][j, l, self.outside_label, :] += counts

                for m in range(nscores):
                    counts = (C[:, l:l+1][1:, :] * (1 - doc_start[1:]) * C[:, m:m+1][:-1, :]).T.dot(Tj[1:]).reshape(-1)
                    self.alpha_taggers[model_idx][j, l, m, :] += counts

    def read_lnPi(self, l, C, Cprev, doc_id, Krange, nscores, blanks):
        # if there is a doc start, Cprev should contain the outside_label value or a -1 for a missing value.
        # In seq.py, this is handled already.

        if l is None:
            if np.isscalar(Krange):
                Krange = np.array([Krange])[None, :]
            blanks_prev = Cprev == -1
            if np.any(blanks_prev):
                Cprev = np.copy(Cprev)
                Cprev[Cprev == -1] = self.outside_label

            result = self.lnPi[:, C, Cprev, Krange]
            result[:, blanks] = 0
        else:
            blanks_prev = Cprev == -1
            if np.any(blanks_prev):
                Cprev = np.copy(Cprev)
                Cprev[Cprev == -1] = self.outside_label

            result = self.lnPi[l, C, Cprev, Krange]
            result[blanks] = 0

        return np.sum(result, axis=-1)

    def read_lnPi_taggers(self, l, C, Cprev, nscores, model_idx):
        if l is None:
            return self.lnPi_taggers[model_idx][:, C, Cprev, 0]
        else:
            return self.lnPi_taggers[model_idx][l, C, Cprev, 0]

    def expand_alpha0(self, C, K, doc_start, nscores):
        '''
        Take the alpha0 for one worker and expand.
        :return:
        '''
        L = self.alpha0.shape[0]

        # set priors
        if self.alpha0 is None:
            # dims: true_label[t], current_anno c[t],  previous_anno c[t-1], annotator k
            self.alpha0 = np.ones((L, nscores, self.num_prev_labels, K)) + 1.0 * np.eye(L)[:, :, None, None]
        else:
            self.alpha0 = self.alpha0[:, :, None, None]
            self.alpha0 = np.tile(self.alpha0, (1, 1, self.num_prev_labels, K))

        for midx in range(self.nModels):
            if self.alpha0_taggers[midx] is None:
                self.alpha0_taggers[midx] = np.ones((L, L, L, 1)) + 1.0 * np.eye(L)[:, :, None, None]
            elif self.alpha0_taggers[midx].ndim == 2:
                self.alpha0_taggers[midx] = self.alpha0_taggers[midx][:, :, None, None]
                self.alpha0_taggers[midx] = np.tile(self.alpha0_taggers[midx], (1, 1, L, 1))

        if self.tagging_scheme == 'IOB':
            restricted_labels = self.beginning_labels  # labels that cannot follow an outside label
            unrestricted_labels = self.inside_labels  # labels that can follow any label
        elif self.tagging_scheme == 'IOB2':
            restricted_labels = self.inside_labels
            unrestricted_labels = self.beginning_labels
        else:  # no special labels
            restricted_labels = []
            unrestricted_labels = []

        # The outside labels often need a stronger bias because they are way more frequent
        self.alpha0[self.beginning_labels, self.beginning_labels] *= self.alpha0_B_factor
        # self.alpha0[self.outside_label, self.outside_label] *= self.alpha0_B_factor

        # set priors for pseudo-counts invalid transitions (to low values)
        for i, restricted_label in enumerate(restricted_labels):
            # Not allowed: from outside to inside
            disallowed_count = self.alpha0[:, restricted_label, self.outside_label] - self.rare_transition_pseudocount

            self.alpha0[:, unrestricted_labels[i], self.outside_label] += disallowed_count  # previous expts use this
            self.alpha0[:, restricted_label, self.outside_label] = self.rare_transition_pseudocount

            for midx in range(self.nModels):
                disallowed_count = self.alpha0_taggers[midx][:, restricted_label, self.outside_label] \
                                   - self.rare_transition_pseudocount
                self.alpha0_taggers[midx][:, unrestricted_labels[i], self.outside_label] += disallowed_count
                self.alpha0_taggers[midx][:, restricted_label, self.outside_label] = self.rare_transition_pseudocount

            # Ban jumps from a B of one type to an I of another type
            for typeid, other_unrestricted_label in enumerate(unrestricted_labels):
                # prevent transitions from unrestricted to restricted if they don't have the same types
                if typeid == i:  # same type is allowed
                    continue

                # *** These seem to have a positive effect
                disallowed_count = self.alpha0[:, restricted_label, other_unrestricted_label] \
                                   - self.rare_transition_pseudocount
                # # pseudocount is (alpha0 - 1) but alpha0 can be < 1. Removing the pseudocount maintains the relative
                # weights between label values
                self.alpha0[:, restricted_labels[typeid], other_unrestricted_label] += disallowed_count
                # set the disallowed transition to as close to zero as possible
                self.alpha0[:, restricted_label, other_unrestricted_label] = self.rare_transition_pseudocount

                for midx in range(self.nModels):
                    disallowed_count = self.alpha0_taggers[midx][:, restricted_label, other_unrestricted_label, :] \
                                       - self.rare_transition_pseudocount
                    self.alpha0_taggers[midx][:, restricted_labels[typeid], other_unrestricted_label] \
                        += disallowed_count # sticks to wrong type
                    self.alpha0_taggers[midx][:, restricted_label, other_unrestricted_label] \
                        = self.rare_transition_pseudocount

            # Ban jumps between Is of different types
            for typeid, other_restricted_label in enumerate(restricted_labels):
                if typeid == i:
                    continue

                # set the disallowed transition to as close to zero as possible
                # disallowed_count = self.alpha0[:, restricted_label, other_restricted_label, :]
                # - self.rare_transition_pseudocount
                # self.alpha0[:, other_restricted_label, other_restricted_label, :] += disallowed_count #TMP
                self.alpha0[:, restricted_label, other_restricted_label] = self.rare_transition_pseudocount

                for midx in range(self.nModels):
                    # disallowed_count = self.alpha0_taggers[midx][:, restricted_label, other_restricted_label, :]
                    # - self.rare_transition_pseudocount
                    # self.alpha0_taggers[midx][:, other_restricted_label, other_restricted_label, :]
                    # += disallowed_count  # TMP sticks to wrong type
                    self.alpha0_taggers[midx][:, restricted_label, other_restricted_label] \
                        = self.rare_transition_pseudocount

    def _calc_EPi(self, alpha):
        return alpha / np.sum(alpha, axis=1)[:, None, :, :]

    def lowerbound_terms(self):
        # the dimension over which to sum, i.e. over which the values are parameters of a single Dirichlet
        sum_dim = 1 # in the case that we have multiple Dirichlets per worker, e.g. IBCC, sequential-BCC model

        lnpPi = log_dirichlet_pdf(self.alpha0, self.lnPi, sum_dim)
        lnqPi = log_dirichlet_pdf(self.alpha, self.lnPi, sum_dim)

        for midx, alpha0_taggers in enumerate(self.alpha0_taggers):
            lnpPi += log_dirichlet_pdf(alpha0_taggers, self.lnPi_taggers[midx], sum_dim)
            lnqPi += log_dirichlet_pdf(self.alpha_taggers[midx], self.lnPi_taggers[midx], sum_dim)

        return lnpPi, lnqPi