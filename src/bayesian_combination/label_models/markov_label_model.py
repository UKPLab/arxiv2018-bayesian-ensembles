import numpy as np
from joblib.parallel import delayed

from baselines.dawid_and_skene import ds
from bayesian_combination.annotator_models.annotator_model import log_dirichlet_pdf
from bayesian_combination.label_models.label_model import LabelModel, _doc_forward_pass, _doc_backward_pass, \
    _doc_most_probable_sequence
from scipy.special import psi, logsumexp

class MarkovLabelModel(LabelModel):
    def __init__(self, beta0, L, verbose, tagging_scheme,
                 beginning_labels, inside_labels, outside_label):

        rare_transition_pseudocount = 1e-12

        self.C_by_doc = None
        self.lnR_ = None

        self.outside_label = outside_label

        super().__init__(beta0, L, verbose)

        if self.beta0.ndim != 2:
            return

        # set priors for invalid transitions (to low values)
        if tagging_scheme == 'IOB':
            restricted_labels = beginning_labels  # labels that cannot follow an outside label
            unrestricted_labels = inside_labels  # labels that can follow any label
        elif tagging_scheme == 'IOB2':
            restricted_labels = inside_labels
            unrestricted_labels = beginning_labels
        else:  # no special tagging scheme, don't set any priors related to transition rules
            restricted_labels = []
            unrestricted_labels = []
            if outside_label == -1:  # have to restrict the prior state
                self.beta0[:, outside_label] = 0


        for i, restricted_label in enumerate(restricted_labels):
            # pseudo-counts for the transitions that are not allowed from outside to inside
            disallowed_counts = self.beta0[outside_label, restricted_label] - rare_transition_pseudocount
            # self.beta0[outside_label, unrestricted_labels[i]] += disallowed_counts
            self.beta0[outside_label, outside_label] += disallowed_counts
            self.beta0[outside_label, restricted_label] = rare_transition_pseudocount

            # cannot jump from one type to another
            for j, unrestricted_label in enumerate(unrestricted_labels):
                if i == j:
                    continue  # this transitiion is allowed
                disallowed_counts = self.beta0[unrestricted_label, restricted_label] - rare_transition_pseudocount
                self.beta0[unrestricted_label, restricted_labels[j]] += disallowed_counts
                self.beta0[unrestricted_label, restricted_label] = rare_transition_pseudocount

            # can't switch types mid annotation
            for j, other_restricted_label in enumerate(restricted_labels):
                if other_restricted_label == restricted_label:
                    continue
                # disallowed_counts = self.beta0[other_restricted_label, restricted_label] - rare_transition_pseudocount
                # self.beta0[other_restricted_label, other_restricted_label] += disallowed_counts
                self.beta0[other_restricted_label, restricted_label] = rare_transition_pseudocount

    def update_B(self, features, ln_features_likelihood, predict_only=False):
        '''
        Update the transition model.
        '''
        self.beta = self.beta0 + np.sum(self.Et_joint, 0)

        self.lnB = psi(self.beta) - psi(np.sum(self.beta, -1))[:, None]

        self.lnB = self.lnB[None, :, :] + ln_features_likelihood[:, None, :]

        if np.any(np.isnan(self.lnB)):
            print('update_B: nan value encountered!')

    def init_t(self, C, doc_start, A, blanks, gold=None):

        super().init_t(C, doc_start, A, blanks, gold)

        B = self.beta0 / np.sum(self.beta0, axis=1)[:, None]

        # initialise using Dawid and Skene
        self.Et = ds(C, self.L, 0.1)

        self.Et_joint = np.zeros((self.N, self.L, self.L))
        for l in range(self.L):
            self.Et_joint[1:, l, :] = self.Et[:-1, l][:, None] * self.Et[1:, :] * B[l, :][None, :]

        self.Et_joint[self.ds, :, :] = 0
        self.Et_joint[self.ds, self.outside_label, :] = self.Et[self.ds, :] * B[self.outside_label, :]

        self.Et_joint /= np.sum(self.Et_joint, axis=(1, 2))[:, None, None]

        # # initialise using the fraction of votes for each label
        # self.Et_joint = np.zeros((self.N, self.L, self.L))
        # for l in range(self.L):
        #     for m in range(self.L):
        #         self.Et_joint[1:, l, m] = np.sum((C[:-1]==l) & (C[1:]==m), axis=1)
        # self.Et_joint += self.beta0[None, :, :]
        # self.Et_joint[doc_start, :, :] = 0
        # for l in range(self.L):
        #     self.Et_joint[doc_start, self.outside_label, l] = np.sum(C[doc_start, :]==l, axis=1) \
        #                                                       + self.beta0[self.outside_label, l]
        # self.Et_joint /= np.sum(self.Et_joint, axis=(1, 2))[:, None, None]
        #
        # self.Et = np.sum(self.Et_joint, axis=1)

        return self.Et

    def update_t(self, parallel, C_data):

        self.parallel = parallel
        self.Cdata = C_data

        # calculate alphas and betas using forward-backward bayesian_combination
        self._parallel_forward_pass()

        if self.verbose:
            print("BAC iteration %i: completed forward pass" % self.iter)

        self._parallel_backward_pass()
        if self.verbose:
            print("BAC iteration %i: completed backward pass" % self.iter)

        # update q_t and q_t_joint
        self._expec_joint_t()

        self.Et = np.sum(self.Et_joint, axis=1)

        if self.gold is not None:
            goldidxs = self.gold != -1
            self.Et[goldidxs, :] = 0
            self.Et[goldidxs, self.gold[goldidxs]] = 1.0

            self.Et_joint[goldidxs, :, :] = 0
            if not hasattr(self, 'goldprev'):
                self.goldprev = np.append([self.outside_label], self.gold[:-1])[goldidxs]

            self.Et_joint[goldidxs, self.goldprev, self.gold[goldidxs]] = 1.0

        return self.Et

    def _parallel_forward_pass(self):
        '''
        Perform the forward pass of the Forward-Backward bayesian_combination (for multiple documents in parallel).
        '''
        # split into documents
        if self.C_by_doc is None:
            self.splitidxs = np.where(self.ds)[0][1:]
            self.C_by_doc = np.split(self.C, self.splitidxs, axis=0)
            # self.Cprev_by_doc = np.split(self.Cprev, self.splitidxs, axis=0)
            self.blanks_by_doc = np.split(self.blanks, self.splitidxs, axis=0)

        self.lnB_by_doc = np.split(self.lnB, self.splitidxs, axis=0)

        # print('Complete size of C = %s and lnB = %s' % (str(C.shape), str(lnB.shape)))

        self.C_data_by_doc = []

        # run forward pass for each doc concurrently
        # option backend='threading' does not work here because of the shared objects locking. Can we release them read-only?
        if self.Cdata is not None and len(self.Cdata) > 0:
            for m, Cdata_m in enumerate(self.Cdata):
                self.C_data_by_doc.append(np.split(Cdata_m, self.splitidxs, axis=0))

            if len(self.Cdata) >= 1:
                self.C_data_by_doc = list(zip(*self.C_data_by_doc))

            res = self.parallel(delayed(_doc_forward_pass)(d, C_doc, self.C_data_by_doc[d], self.blanks_by_doc[d],
                                                           self.lnB_by_doc[d],
                                                           self.A, self.outside_label)
                           for d, C_doc in enumerate(self.C_by_doc))
        else:
            res = self.parallel(delayed(_doc_forward_pass)(d, C_doc, None, self.blanks_by_doc[d], self.lnB_by_doc[d],
                                                           self.A, self.outside_label)
                           for d, C_doc in enumerate(self.C_by_doc))

        # reformat results
        self.lnR_ = np.concatenate([r[0] for r in res], axis=0)
        self.scaling = [r[1] for r in res]


    def _parallel_backward_pass(self):
        '''
        Perform the backward pass of the Forward-Backward bayesian_combination (for multiple documents in parallel).
        '''
        if self.Cdata is not None and len(self.Cdata) > 0:
            res = self.parallel(delayed(_doc_backward_pass)(d, doc, self.C_data_by_doc[d], self.blanks_by_doc[d],
                                                            self.lnB_by_doc[d], self.L,
                                                            self.A, self.scaling[d]
                                                    ) for d, doc in enumerate(self.C_by_doc))
        else:
            res = self.parallel(delayed(_doc_backward_pass)(d, doc, None, self.blanks_by_doc[d],
                                                            self.lnB_by_doc[d], self.L, self.A, self.scaling[d],
                                                    ) for d, doc in enumerate(self.C_by_doc))

        # reformat results
        self.lnLambda = np.concatenate(res, axis=0)


    def _expec_joint_t(self):
        '''
        Calculate joint label probabilities for each pair of tokens.
        '''
        # initialise variables
        L = self.lnB.shape[-1]
        K = self.C.shape[-1]

        lnS = np.repeat(self.lnLambda[:, None, :], L, 1)

        # flags to indicate whether the entries are valid or should be zeroed out later.
        flags = -np.inf * np.ones_like(lnS)
        flags[np.where(self.ds == 1)[0], self.outside_label, :] = 0
        flags[np.where(self.ds == 0)[0], :, :] = 0

        for l in range(L):

            # Non-doc-starts
            lnPi_terms = self.A.read_lnPi(l, self.C, self.Cprev, None, np.arange(K)[None, :], self.L, self.blanks)
            if self.Cdata is not None and len(self.Cdata):
                lnPi_data_terms = np.zeros(self.Cdata[0].shape[0], dtype=float)

                for model_idx in range(len(self.Cdata)):
                    C_data_prev = np.append(np.zeros((1, L), dtype=int), self.Cdata[model_idx][:-1, :], axis=0)
                    C_data_prev[self.ds, :] = 0
                    C_data_prev[self.ds, self.outside_label] = 1
                    C_data_prev[0, self.outside_label] = 1

                    for m in range(L):
                        weights = self.Cdata[model_idx][:, m:m + 1] * C_data_prev

                        for n in range(L):
                            lnPi_data_terms_mn = weights[:, n] * self.A.read_lnPi_taggers(l, m, n, self.L, model_idx)
                            lnPi_data_terms_mn[np.isnan(lnPi_data_terms_mn)] = 0
                            lnPi_data_terms += lnPi_data_terms_mn
            else:
                lnPi_data_terms = 0

            loglikelihood_l = (lnPi_terms + lnPi_data_terms)[:, None]

            # for the other times. The document starts will get something invalid written here too, but it will be killed off by the flags
            # print('_expec_joint_t_quick: For class %i: adding the R terms from previous data points' % l)
            lnS[:, :, l] += loglikelihood_l + self.lnB[:, :, l]
            lnS[1:, :L, l] += self.lnR_[:-1, :]

        # print('_expec_joint_t_quick: Normalising...')

        # normalise and return
        lnS = lnS + flags
        if np.any(np.isnan(np.exp(lnS))):
            print('_expec_joint_t: nan value encountered (1) ')

        lnS = lnS - logsumexp(lnS, axis=(1, 2))[:, None, None]

        if np.any(np.isnan(np.exp(lnS))):
            print('_expec_joint_t: nan value encountered')

        self.Et_joint = np.exp(lnS)


    def most_likely_labels(self, word_ll):
        '''
        Use Viterbi decoding to ensure we make a valid prediction. There
        are some cases where most probable sequence does not match argmax(self.q_t,1). E.g.:
        [[0.41, 0.4, 0.19], [[0.41, 0.42, 0.17]].
        Taking the most probable labels results in an illegal sequence of [0, 1].
        Using most probable sequence would result in [0, 0].
        '''
        if self.beta.ndim >= 2 and self.beta.shape[0] > 1:
            EB = self.beta / np.sum(self.beta, axis=1)[:, None]
        else:
            EB = self.beta / np.sum(self.beta)
            EB = np.tile(EB[None, :], (self.L, 1))

        lnEB = np.zeros_like(EB)
        lnEB[EB != 0] = np.log(EB[EB != 0])
        lnEB[EB == 0] = -np.inf

        # update the expected log word likelihoods
        if word_ll is None:
            lnEB = np.tile(lnEB[None, :, :], (self.C.shape[0], 1, 1))
        else:
            lnEB = lnEB[None, :, :] + word_ll[:, None, :]

        # split into documents
        lnEB_by_doc = np.split(lnEB, self.splitidxs, axis=0)

        C_data_by_doc = []
        for C_data_m in self.Cdata:
            C_data_by_doc.append(np.split(C_data_m, self.splitidxs, axis=0))
        if len(self.Cdata) >= 1:
            C_data_by_doc = list(zip(*C_data_by_doc))

            res = self.parallel(delayed(_doc_most_probable_sequence)(d, doc, C_data_by_doc[d], lnEB_by_doc[d], self.L,
                                                         self.C.shape[1], self.A, self.outside_label)
                           for d, doc in enumerate(self.C_by_doc))
        else:
            res = self.parallel(delayed(_doc_most_probable_sequence)(d, doc, None, lnEB_by_doc[d], self.L,
                                                         self.C.shape[1], self.A, self.outside_label)
                           for d, doc in enumerate(self.C_by_doc))

        # note: we are using ElnPi instead of lnEPi, I think is not strictly correct.

        # reformat results
        pseq = np.array(list(zip(*res))[0])
        seq = np.concatenate(list(zip(*res))[1], axis=0)

        return pseq, seq

    def lowerbound_terms(self):
        lnpCt = (self.lnR_ + np.concatenate(self.scaling, axis=0)[:, None]) * self.Et
        lnpCt = np.sum(lnpCt)

        qt_sum = np.sum(self.Et_joint, axis=2)[:, :, None]
        qt_sum[qt_sum == 0] = 1.0 # doesn't matter, they will be multiplied by zero. This avoids the warning
        q_t_cond = self.Et_joint / qt_sum
        q_t_cond[q_t_cond == 0] = 1.0 # doesn't matter, they will be multiplied by zero. This avoids the warning
        lnqt = self.Et_joint * np.log(q_t_cond)
        lnqt[np.isinf(lnqt) | np.isnan(lnqt)] = 0
        lnqt = np.sum(lnqt)

        lnB = psi(self.beta) - psi(np.sum(self.beta, 1))[:, None]
        lnpB = log_dirichlet_pdf(self.beta0, lnB, 1)
        lnqB = log_dirichlet_pdf(self.beta, lnB, 1)
        lnpB = np.sum(lnpB)
        lnqB = np.sum(lnqB)

        return lnpCt + lnpB, lnqt + lnqB