'''
Bayesian sequence combination, variational Bayes implementation.
'''
import datetime

import numpy as np
from scipy.sparse.coo import coo_matrix
from scipy.special import logsumexp, psi, gammaln
from scipy.optimize.optimize import fmin
from joblib import Parallel, delayed, cpu_count, effective_n_jobs
import warnings

#from lample_lstm_tagger.loader import tag_mapping
from baselines.dawid_and_skene import ds
from bsc.acc import AccuracyWorker
from bsc.cm import ConfusionMatrixWorker
from bsc.cv import VectorWorker
from bsc.indfeatures import IndependentFeatures
from bsc.lstm import LSTM
from bsc.mace import MACEWorker
from bsc.seq import SequentialWorker


class BSC(object):

    K = None  # number of annotators
    L = None  # number of class labels

    nscores = None  # number of possible values a token can have, usually this is L + 1 (add one to account for unannotated tokens)

    beta0 = None  # ground truth priors

    lnB = None  # transition matrix
    lnPi = None  # worker confusion matrices

    q_t = None  # current true label estimates
    q_t_old = None  # previous true labels estimates

    iter = 0  # current iteration

    max_iter = None  # maximum number of iterations
    eps = None  # maximum difference of estimate differences in convergence chack

    def __init__(self, L=3, K=5, max_iter=20, eps=1e-4, inside_labels=[0], outside_labels=[1, -1], beginning_labels=[2],
                 before_doc_idx=1, alpha0_diags=1.0, alpha0_factor=1.0, alpha0_outside_factor=1.0, beta0_factor=1.0, nu0=1,
                 worker_model='ibcc', data_model=None, tagging_scheme='IOB2', transition_model='HMM', no_words=False,
                 model_dir=None, reload_lstm=False, embeddings_file=None, rare_transition_pseudocount=1e-10):

        self.tagging_scheme = tagging_scheme # may be 'IOB2' (all annotations start with B) or 'IOB' (only annotations
        # that follow another start with B).

        self.L = L
        self.nscores = L
        self.K = K

        self.inside_labels = inside_labels
        self.outside_labels = outside_labels
        self.beginning_labels = beginning_labels

        self.nu0 = nu0

        # choose whether to use the HMM transition model or not
        if transition_model == 'HMM':
            self.beta0 = np.ones((self.L + 1, self.L)) * beta0_factor
            self._update_B = self._update_B_trans
            self.use_ibcc_to_init = True
            self.has_transition_constraints = True
        else:
            self.beta0 = np.ones((self.L)) * beta0_factor
            self._update_B = self._update_B_notrans
            self.use_ibcc_to_init = False
            self.has_transition_constraints = False

        # choose data model
        self.max_data_updates_at_end = 0  # no data model to be updated after everything else has converged.

        self.no_words = no_words

        if data_model is None:
            self.data_model = []
        else:
            self.data_model = []
            for modelstr in data_model:
                if modelstr == 'LSTM':
                    self.data_model.append(LSTM(model_dir, reload_lstm, embeddings_file))
                    self.max_data_updates_at_end = 3 #20  # allow the other parameters to converge first, then update LSTM
                    # with small number of iterations to avoid overfitting
                if modelstr == 'IF':
                    self.data_model.append(IndependentFeatures())

        # choose type of worker model
        if worker_model == 'acc':
            self.A = AccuracyWorker(alpha0_diags, alpha0_factor, L, len(data_model))
            self.alpha_shape = (2)

        elif worker_model == 'mace':
            self.A = MACEWorker(alpha0_diags, alpha0_factor, L, len(data_model))
            self.alpha_shape = (2 + self.nscores)
            self._lowerbound_pi_terms = self._lowerbound_pi_terms_mace

        elif worker_model == 'ibcc':
            self.A = ConfusionMatrixWorker(alpha0_diags, alpha0_factor, L, len(data_model))
            self.alpha_shape = (self.L, self.nscores)

        elif worker_model == 'seq':
            self.A = SequentialWorker(alpha0_diags, alpha0_factor, L, len(data_model), tagging_scheme,
                            beginning_labels, inside_labels, outside_labels, alpha0_outside_factor,
                            rare_transition_pseudocount)
            self.alpha_shape = (self.L, self.nscores)

        elif worker_model == 'vec':
            self.A = VectorWorker(alpha0_diags, alpha0_factor, L, len(data_model))
            self.alpha_shape = (self.L, self.nscores)

        self.alpha0_outside_factor = alpha0_outside_factor

        self.rare_transition_pseudocount = rare_transition_pseudocount

        self.before_doc_idx = before_doc_idx  # identifies which true class value is assumed for the label before the start of a document

        self.max_iter = max_iter #- self.max_data_updates_at_end  # maximum number of iterations before training data models
        self.eps = eps  # threshold for convergence
        self.iter = 0
        self.max_internal_iters = 20

        self.verbose = False  # can change this if you want progress updates to be printed


    def _set_transition_constraints(self):

        if self.beta0.ndim != 2:
            return

        # set priors for invalid transitions (to low values)
        if self.tagging_scheme == 'IOB':
            restricted_labels = self.beginning_labels  # labels that cannot follow an outside label
            unrestricted_labels = self.inside_labels  # labels that can follow any label
        elif self.tagging_scheme == 'IOB2':
            restricted_labels = self.inside_labels
            unrestricted_labels = self.beginning_labels

        for i, restricted_label in enumerate(restricted_labels):
            # pseudo-counts for the transitions that are not allowed from outside to inside
            disallowed_counts = self.beta0[self.outside_labels, restricted_label] - self.rare_transition_pseudocount
            # self.beta0[self.outside_labels, self.beginning_labels[i]] += disallowed_counts
            self.beta0[self.outside_labels, self.outside_labels[0]] += disallowed_counts
            self.beta0[self.outside_labels, restricted_label] = self.rare_transition_pseudocount

            # cannot jump from one type to another
            for j, unrestricted_label in enumerate(unrestricted_labels):
                if i == j:
                    continue  # this transitiion is allowed
                disallowed_counts = self.beta0[unrestricted_label, restricted_label] - self.rare_transition_pseudocount
                self.beta0[unrestricted_label, restricted_labels[j]] += disallowed_counts
                self.beta0[unrestricted_label, restricted_label] = self.rare_transition_pseudocount

            # can't switch types mid annotation
            for j, other_restricted_label in enumerate(restricted_labels):
                if other_restricted_label == restricted_label:
                    continue
                self.beta0[other_restricted_label, restricted_label] = self.rare_transition_pseudocount


    def _initB(self):
        self.beta = self.beta0

        if self.beta0.ndim >= 2:
            beta0_sum = psi(np.sum(self.beta0, -1))[:, None]
        else:
            beta0_sum = psi(np.sum(self.beta0))

        self.lnB = psi(self.beta0) - beta0_sum

        if self.beta0.ndim == 1:
            self.lnB = np.tile(self.lnB, (self.L + 1, 1))

        self.lnB = np.tile(self.lnB, (self.N, 1, 1))

    def _lowerbound_pi_terms_mace(self, alpha0, alpha, lnPi):
        # the dimension over which to sum, i.e. over which the values are parameters of a single Dirichlet
        sum_dim = 0 # in the case that we have multiple Dirichlets per worker, e.g. IBCC, sequential-BCC model

        lnpPi_correct = _log_dir(alpha0[0:2, :], lnPi[0:2, :], sum_dim)
        lnpPi_strategy = _log_dir(alpha0[2:, :], lnPi[2:, :], sum_dim)

        lnqPi_correct = _log_dir(alpha[0:2, :], lnPi[0:2, :], sum_dim)
        lnqPi_strategy = _log_dir(alpha[2:, :], lnPi[2:, :], sum_dim)

        return lnpPi_correct + lnpPi_strategy, lnqPi_correct + lnqPi_strategy

    def _lowerbound_pi_terms(self, alpha0, alpha, lnPi):
        # the dimension over which to sum, i.e. over which the values are parameters of a single Dirichlet
        if alpha.ndim == 2:
            sum_dim = 0 # in the case that we have only one Dirichlet per worker, e.g. accuracy model
        else:
            sum_dim = 1 # in the case that we have multiple Dirichlets per worker, e.g. IBCC, sequential-BCC model

        lnpPi = _log_dir(alpha0, lnPi, sum_dim)

        lnqPi = _log_dir(alpha, lnPi, sum_dim)

        return lnpPi, lnqPi

    def _lnpt(self):
        lnpt = self.lnB.copy()
        lnpt[np.isinf(lnpt) | np.isnan(lnpt)] = 0
        lnpt = lnpt * self.q_t_joint

        return lnpt

    def lowerbound(self):
        '''
        Compute the variational lower bound on the log marginal likelihood.
        '''
        lnp_features_and_Cdata = 0
        lnq_Cdata = 0

        for model in self.data_model:
            lnp_features_and_Cdata += model.log_likelihood(model.C_data, self.q_t)
            lnq_Cdata_m = model.C_data * np.log(model.C_data)
            lnq_Cdata_m[model.C_data == 0] = 0
            lnq_Cdata += lnq_Cdata_m

        lnq_Cdata = np.sum(lnq_Cdata)

        lnpC = 0
        C = self.C.astype(int)
        C_prev = np.concatenate((np.zeros((1, C.shape[1]), dtype=int), C[:-1, :]))
        C_prev[self.doc_start.flatten() == 1, :] = 0
        C_prev[C_prev == 0] = self.before_doc_idx + 1  # document starts or missing labels
        valid_labels = (C != 0).astype(float)

        for j in range(self.L):
            # self.lnPi[j, C - 1, C_prev - 1, np.arange(self.K)[None, :]]
            lnpCj = valid_labels * self.A._read_lnPi(j, C - 1, C_prev - 1, np.arange(self.K)[None, :], self.nscores) \
                    * self.q_t[:, j:j+1]
            lnpC += lnpCj

        lnpt = self._lnpt()

        lnpCt = np.sum(lnpC) + np.sum(lnpt)

        # trying to handle warnings
        qt_sum = np.sum(self.q_t_joint, axis=2)[:, :, None]
        qt_sum[qt_sum==0] = 1.0 # doesn't matter, they will be multiplied by zero. This avoids the warning
        q_t_cond = self.q_t_joint / qt_sum
        q_t_cond[q_t_cond == 0] = 1.0 # doesn't matter, they will be multiplied by zero. This avoids the warning
        lnqt = self.q_t_joint * np.log(q_t_cond)
        lnqt[np.isinf(lnqt) | np.isnan(lnqt)] = 0
        lnqt = np.sum(lnqt)
        warnings.filterwarnings('always')

        # E[ln p(\pi | \alpha_0)]
        # E[ln q(\pi)]
        lnpPi, lnqPi = self._lowerbound_pi_terms(self.alpha0, self.alpha, self.lnPi)

        for model in self.data_model:
            lnpPi_model, lnqPi_model = self._lowerbound_pi_terms(model.alpha0_data,
                                                                 model.alpha_data,
                                                                 model.lnPi)

            lnpPi += lnpPi_model
            lnqPi += lnqPi_model

        # E[ln p(A | nu_0)]
        # TODO this is wrong since lnB is now transition matrix + word observation likelihood
        x = (self.beta0 - 1) * self.lnB
        gammaln_nu0 = gammaln(self.beta0)
        invalid_nus = np.isinf(gammaln_nu0) | np.isinf(x) | np.isnan(x)
        gammaln_nu0[invalid_nus] = 0
        x[invalid_nus] = 0
        x = np.sum(x, axis=1)
        z = gammaln(np.sum(self.beta0, 1)) - np.sum(gammaln_nu0, 1)
        z[np.isinf(z)] = 0
        lnpA = np.sum(x + z)

        # E[ln q(A)]
        # TODO this is wrong since lnB is now transition matrix + word observation likelihood
        x = (self.beta - 1) * self.lnB
        x[np.isinf(x)] = 0
        gammaln_nu = gammaln(self.beta)
        gammaln_nu[invalid_nus] = 0
        x[invalid_nus] = 0
        x = np.sum(x, axis=1)
        z = gammaln(np.sum(self.beta, 1)) - np.sum(gammaln_nu, 1)
        z[np.isinf(z)] = 0
        lnqA = np.sum(x + z)

        print('Computing LB: %f, %f, %f, %f, %f, %f, %f, %f' % (lnpCt, lnqt, lnp_features_and_Cdata, lnq_Cdata,
                                                                lnpPi, lnqPi, lnpA, lnqA))
        lb = lnpCt - lnqt + lnp_features_and_Cdata - lnq_Cdata + lnpPi - lnqPi + lnpA - lnqA
        return lb

    def optimize(self, C, doc_start, features=None, maxfun=50, converge_workers_first=False):
        '''
        Run with MLII optimisation over the lower bound on the log-marginal likelihood.
        Optimizes the confusion matrix prior to the same values for all previous labels, and the scaling of the transition matrix
        hyperparameters.
        '''
        self.opt_runs = 0

        def neg_marginal_likelihood(hyperparams, C, doc_start):
            # set hyperparameters

            n_alpha_elements = len(hyperparams) - 1

            self.alpha0 = np.exp(hyperparams[0:n_alpha_elements]).reshape(self.alpha_shape)
            self.beta0 = np.ones((self.L + 1, self.L)) * np.exp(hyperparams[-1])

            # run the method
            self.run(C, doc_start, features, converge_workers_first=converge_workers_first)

            # compute lower bound
            lb = self.lowerbound()

            print("Run %i. Lower bound: %.5f, alpha_0 = %s, nu0 scale = %.3f" % (self.opt_runs,
                                                 lb, str(np.exp(hyperparams[0:-1])), np.exp(hyperparams[-1])))

            self.opt_runs += 1
            return -lb

        initialguess = np.log(np.append(self.alpha0.flatten(), self.beta0[0, 0]))
        ftol = 1.0  #1e-3
        opt_hyperparams, _, _, _, _ = fmin(neg_marginal_likelihood, initialguess, args=(C, doc_start), maxfun=maxfun,
                                                     full_output=True, ftol=ftol, xtol=1e100)

        print("Optimal hyper-parameters: alpha_0 = %s, nu0 scale = %s" % (np.array2string(np.exp(opt_hyperparams[:-1])),
                                                                          np.array2string(np.exp(opt_hyperparams[-1]))))

        C_data = []
        for model in self.data_model:
            C_data.append(model.C_data)

        return self.q_t, self._most_probable_sequence(C, C_data, doc_start, self.features)[1]


    def _init_t(self, C, doc_start):
        C = C-1
        doc_start = doc_start.flatten()

        if self.beta0.ndim == 2:
            B = self.beta0 / np.sum(self.beta0, axis=1)[:, None]
            beta0 = np.mean(self.beta0, 0)
        else:
            B = np.tile((self.beta0 / np.sum(self.beta0))[None, :], (self.L, 1))
            beta0 = self.beta0

        if self.use_ibcc_to_init:
            self.q_t = ds(C, self.L, 0.1)
        else:
            self.q_t = np.zeros((C.shape[0], self.L)) + beta0
            for l in range(self.L):
                self.q_t[:, l] += np.sum(C == l, axis=1)
            self.q_t /= np.sum(self.q_t, axis=1)[:, None]

        self.q_t_joint = np.zeros((self.N, self.L+1, self.L))
        for l in range(self.L):
            self.q_t_joint[1:, l, :] =  self.q_t[:-1, l][:, None] * self.q_t[1:, :] * B[l, :][None, :]

        self.q_t_joint[doc_start, :, :] = 0
        self.q_t_joint[doc_start, self.before_doc_idx, :] = self.q_t[doc_start, :] * B[self.before_doc_idx, :]

        self.q_t_joint /= np.sum(self.q_t_joint, axis=(1,2))[:, None, None]


    def _update_t(self, parallel, C, C_data, doc_start):

        # calculate alphas and betas using forward-backward bsc
        self.lnR_ = _parallel_forward_pass(parallel, C, C_data,
                                           self.lnB[:, 0:self.L, :],
                                           doc_start, self.nscores, self.A, self.before_doc_idx)

        if self.verbose:
            print("BAC iteration %i: completed forward pass" % self.iter)

        self.lnLambd = _parallel_backward_pass(parallel, C, C_data,
                                          self.lnB[:, 0:self.L, :],
                                          doc_start, self.nscores, self.A, self.before_doc_idx)
        if self.verbose:
            print("BAC iteration %i: completed backward pass" % self.iter)

        # update q_t and q_t_joint
        self.q_t_joint = _expec_joint_t(self.lnR_, self.lnLambd, self.lnB, C, C_data,
                                        doc_start, self.nscores, self.A, self.blanks, self.before_doc_idx)

        self.q_t = np.sum(self.q_t_joint, axis=1)

        if self.gold is not None:
            goldidxs = self.gold != -1
            self.q_t[goldidxs, :] = 0
            self.q_t[goldidxs, self.gold[goldidxs]] = 1.0

            self.q_t_joint[goldidxs, :, :] = 0
            if not hasattr(self, 'goldprev'):
                self.goldprev = np.append([self.before_doc_idx], self.gold[:-1])[goldidxs]

            self.q_t_joint[goldidxs, self.goldprev, self.gold[goldidxs]] = 1.0

        return self.q_t

    def _init_words(self, text):

        self.feat_map = {} # map text tokens to indices
        self.features = []  # list of mapped index values for each token

        if self.no_words:
            return

        N = len(text)

        for feat in text.flatten():
            if feat not in self.feat_map:
                self.feat_map[feat] = len(self.feat_map)

            self.features.append( self.feat_map[feat] )

        self.features = np.array(self.features).astype(int)

        # sparse matrix of one-hot encoding, nfeatures x N, where N is number of tokens in the dataset
        self.features_mat = coo_matrix((np.ones(len(text)), (self.features, np.arange(N)))).tocsr()


    def _update_words(self):
        if self.no_words:
            return np.zeros((self.N, self.L))

        self.nu = self.nu0 + self.features_mat.dot(self.q_t) # nfeatures x nclasses

        # update the expected log word likelihoods
        self.ElnRho = psi(self.nu) - psi(np.sum(self.nu, 0)[None, :])

        # get the expected log word likelihoods of each token
        lnptext_given_t = self.ElnRho[self.features, :]
        lnptext_given_t -= logsumexp(lnptext_given_t, axis=1)[:, None]

        return lnptext_given_t # N x nclasses where N is number of tokens/data points

    def _update_B_trans(self):
        '''
        Update the transition model.
        '''
        self.beta = self.beta0 + np.sum(self.q_t_joint, 0)
        self.lnB = psi(self.beta) - psi(np.sum(self.beta, -1))[:, None]

        ln_word_likelihood = self._update_words()
        self.lnB = self.lnB[None, :, :] + ln_word_likelihood[:, None, :]

        if np.any(np.isnan(self.lnB)):
            print('_calc_q_A: nan value encountered!')

    def _update_B_notrans(self):
        '''
        Update the transition model.

        q_t_joint is N x L+1 x L. In this case, we just sum up over all previous label values, so the counts are
        independent of the previous labels.
        '''
        self.beta = self.beta0 + np.sum(np.sum(self.q_t_joint, 0), 0)
        self.lnB = psi(self.beta) - psi(np.sum(self.beta, -1))
        self.lnB = np.tile(self.lnB, (self.L+1, 1))

        ln_word_likelihood = self._update_words()
        self.lnB = self.lnB[None, :, :] + ln_word_likelihood[:, None, :]

        if np.any(np.isnan(self.lnB)):
            print('_calc_q_A: nan value encountered!')

    def run(self, C, doc_start, features=None, converge_workers_first=True, crf_probs=False, dev_sentences=[],
            gold_labels=None, C_data_initial=None):
        '''
        Runs the BAC bsc with the given annotations and list of document starts.

        '''

        if self.verbose:
            print('BSC: run() called with C shape = %s' % str(C.shape))

        # if there are any gold labels, supply a vector or list of values with -1 for the missing ones
        if gold_labels is not None:
            self.gold = np.array(gold_labels).flatten().astype(int)
        else:
            self.gold = None

        # initialise the hyperparameters to correct sizes
        self.A._expand_alpha0(self.K, self.nscores)

        # validate input data
        assert C.shape[0] == doc_start.shape[0]

        # set the correct number of iterations with the LSTM
        # self.max_data_updates_at_end = self.max_iter - 2

        # transform input data to desired format: unannotated tokens represented as zeros
        C = C.astype(int) + 1
        doc_start = doc_start.astype(bool)
        if doc_start.ndim == 1:
            doc_start = doc_start[:, None]

        self.N = doc_start.size

        if self.iter == 0:
            # try without any transition constraints
            self._set_transition_constraints()

            # initialise transition and confusion matrices
            self._init_words(features)
            self._initB()
            self.A._init_lnPi()

            # initialise variables
            self.q_t_old = np.zeros((C.shape[0], self.L))
            self.q_t = np.ones((C.shape[0], self.L))

            #oldlb = -np.inf

            self.doc_start = doc_start
            self.C = C

            self.blanks = C == 0

        # reset the data model guesses to zero for the first iteration after we restart iterative learning
        for model in self.data_model:
            model.init(C.shape[0], features, doc_start, self.L,
                      self.max_data_updates_at_end if converge_workers_first else self.max_iter, crf_probs,
                      dev_sentences, self.A, self.max_internal_iters)
            self.A.qpi_model(model)

        for midx, model in enumerate(self.data_model):
            if C_data_initial is None:
                model.C_data = np.zeros((self.C.shape[0], self.nscores)) #+ (1.0 / self.nscores)
            else:
                model.C_data = C_data_initial[midx]
                print('integrated model initial labels: ' +
                      str(np.unique(np.argmax(C_data_initial[midx], axis=1))))

        print('Parallel can run %i jobs simultaneously, with %i cores' % (effective_n_jobs(), cpu_count()) )

        self.data_model_updated = self.max_data_updates_at_end
        if converge_workers_first:
            for model in self.data_model:
                if type(model) == LSTM:
                    self.data_model_updated = 0

        self.workers_converged = False

        # timestamp = datetime.datetime.now().strftime('started-%Y-%m-%d-%H-%M-%S')

        # main inference loop
        with Parallel(n_jobs=-1) as parallel: #, backend='threading') as parallel:

            while not self._converged() or not self.workers_converged:

                # print status if desired
                if self.verbose:
                    print("BAC iteration %i in progress" % self.iter)

                self.q_t_old = self.q_t
                C_data = [model.C_data for model in self.data_model]

                if self.verbose:
                    print('Updating with C shape = %s' % str(C.shape))

                if self.iter > 0:
                    self._update_t(parallel, C, C_data, doc_start)
                else:
                    self._init_t(C, doc_start)

                if self.verbose:
                    print("BAC iteration %i: computed label sequence probabilities" % self.iter)

                # update E_lnB
                self._update_B()
                if self.verbose:
                    print("BAC iteration %i: updated transition matrix" % self.iter)

                    # print('BAC transition matrix: ')
                    # print(np.around(self.beta / (np.sum(self.beta, -1)[:, None] if self.beta.ndim > 1 else np.sum(self.beta)), 2))
                    #
                    # print('BAC transition matrix params: ')
                    # print(np.around(self.beta[:self.L], 2))

                for midx, model in enumerate(self.data_model):
                    if not converge_workers_first or self.workers_converged:

                        # Update the data model by retraining the integrated task classifier and obtaining its predictions
                        if model.train_type == 'Bayes':
                            model.C_data = model.fit_predict(self.q_t)
                        elif model.train_type == 'MLE':
                            seq = self._most_probable_sequence(C, C_data, doc_start, self.features, parallel)[1]
                            model.C_data = model.fit_predict(seq)

                        if type(model) == LSTM:
                            self.data_model_updated += 1

                        if self.verbose:
                            print("BAC iteration %i: updated feature-based predictions from %s" %
                                  (self.iter, type(model)))


                        print('integrated model predicted the following labels: ' +
                              str(np.unique(np.argmax(model.C_data, axis=1))))

                    if not converge_workers_first or self.workers_converged or C_data_initial is not None:

                        self.A._post_alpha_data(self.q_t, model.C_data, doc_start, self.nscores, self.before_doc_idx)
                        self.A.q_pi_data(midx)

                        if self.verbose:
                            print("BAC iteration %i: updated model for feature-based predictor of type %s" %
                                  (self.iter, str(type(model))) )

                        # if type(model) == LSTM:
                        #    np.save('LSTM_worker_model_%s.npy' % timestamp, model.alpha_data)
                    elif C_data_initial is None: # model is switched off
                        model.C_data[:] = 0


                # update E_lnpi
                self.A._post_alpha(self.q_t, C, doc_start, self.nscores, self.before_doc_idx)
                self.A.q_pi()

                if self.verbose:
                    print("BAC iteration %i: updated worker models" % self.iter)

                # Note: we are not using this to check convergence -- it's only here to check correctness of bsc
                # Can be commented out to save computational costs.
                # lb = self.lowerbound()
                # print('Iter %i, lower bound = %.5f, diff = %.5f' % (self.iter, lb, lb - oldlb))
                # oldlb = lb

                # increase iteration number
                self.iter += 1

            if self.verbose:
                print("BAC iteration %i: computing most probable sequence..." % self.iter)

            C_data = [model.C_data for model in self.data_model]
            pseq, seq = self._most_probable_sequence(C, C_data, doc_start, self.features, parallel)
        if self.verbose:
            print("BAC iteration %i: fitting/predicting complete." % self.iter)

            # print('BAC final transition matrix: ')
            # print(np.around(self.beta / (np.sum(self.beta, -1)[:, None] if self.beta.ndim > 1 else np.sum(self.beta)), 2))

        # Some code for saving the Seq model to a reasonably readable format
        # import pandas as pd
        # betatable = np.around(self.beta / np.sum(self.beta, axis=1)[:, None], 2)
        # betatable = pd.DataFrame(betatable)
        # betatable.to_csv('./beta_%i.csv' % C.shape[0])
        #
        # flatalpha = np.swapaxes(np.around(self.alpha / np.sum(self.alpha, axis=1)[:, None, :, :], 2), 0, -1).swapaxes(-1, 1
        #              ).swapaxes(-1,2).reshape(self.K, self.L*self.L*(self.L + 1))
        #
        # headers = np.unravel_index(np.arange((self.L+1)*self.L*self.L), dims=(self.L, self.L, self.L+1))
        # columns = ["%i_%i_%i" % (headers[0][i], headers[1][i], headers[2][i])
        #            for i in range(len(headers[0]))]
        # flatalpha = pd.DataFrame(flatalpha, columns=columns)
        # flatalpha.to_csv('./alpha_%i.csv' % C.shape[0])

        return self.q_t, seq, pseq

    def _most_probable_sequence(self, C, C_data, doc_start, features, parallel):
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
            EB = np.tile(EB[None, :], (self.L+1, 1))

        lnEB = np.zeros_like(EB)
        lnEB[EB != 0] = np.log(EB[EB != 0])
        lnEB[EB == 0] = -np.inf

        # update the expected log word likelihoods
        if self.no_words:
            lnEB = np.tile(lnEB[None, :, :], (C.shape[0], 1, 1))
        else:
            ERho = self.nu / np.sum(self.nu, 0)[None, :] # nfeatures x nclasses
            lnERho = np.zeros_like(ERho)
            lnERho[ERho != 0] = np.log(ERho[ERho != 0])
            lnERho[ERho == 0] = -np.inf
            word_ll = lnERho[features, :]

            lnEB = lnEB[None, :, :] + word_ll[:, None, :]

        EPi = self.A.EPi()
        lnEPi = np.zeros_like(EPi)
        lnEPi[EPi != 0] = np.log(EPi[EPi != 0])
        lnEPi[EPi == 0] = -np.inf

        lnEPi_data = []

        for midx, model in enumerate(self.data_model):
            EPi_m = self.A.EPi_data(midx)
            lnEPi_m = np.zeros_like(EPi_m)
            lnEPi_m[EPi_m != 0] = np.log(EPi_m[EPi_m != 0])
            lnEPi_m[EPi_m == 0] = -np.inf
            lnEPi_data.append(lnEPi_m)

        # split into documents
        docs = np.split(C, np.where(doc_start == 1)[0][1:], axis=0)
        lnEB_by_doc = np.split(lnEB, np.where(doc_start == 1)[0][1:], axis=0)

        C_data_by_doc = []
        for C_data_m in C_data:
            C_data_by_doc.append(np.split(C_data_m, np.where(doc_start == 1)[0][1:], axis=0))
        if len(self.data_model) >= 1:
            C_data_by_doc = list(zip(*C_data_by_doc))

        # docs = np.split(C, np.where(doc_start == 1)[0][1:], axis=0)
        # run forward pass for each doc concurrently
        if len(self.data_model):
            res = parallel(delayed(_doc_most_probable_sequence)(doc, C_data_by_doc[d], lnEB_by_doc[d], lnEPi,
                        lnEPi_data, self.L, self.nscores, self.K, self.A, self.before_doc_idx)
                           for d, doc in enumerate(docs))
        else:
            # res = parallel(delayed(_doc_most_probable_sequence)(doc, None, lnEB_by_doc[d], lnEPi, lnEPi_data, self.L,
            #             self.nscores, self.K, self.A, self.before_doc_idx)
            #                for d, doc in enumerate(docs))

            res = [_doc_most_probable_sequence(doc, None, lnEB_by_doc[d], self.L,
                        self.nscores, self.K, self.A, self.before_doc_idx)
                       for d, doc in enumerate(docs)]

        # reformat results
        pseq = np.array(list(zip(*res))[0])
        seq = np.concatenate(list(zip(*res))[1], axis=0)

        return pseq, seq

    def predict(self, doc_start, text):

        with Parallel(n_jobs=-1, backend='threading') as parallel:

            doc_start = doc_start.astype(bool)
            C = np.zeros((len(doc_start), self.K), dtype=int)  # all blank
            self.blanks = C == 0

            self.lnB = psi(self.beta) - psi(np.sum(self.beta, -1))[:, None]

            if self.no_words:
                lnptext_given_t = np.zeros((len(doc_start), self.L))
                features = None
            else:
                # get the expected log word likelihoods of each token
                features = []  # list of mapped index values for each token
                available = []
                for feat in text.flatten():
                    if feat not in self.feat_map:
                        available.append(0)
                        features.append(0)
                    else:
                        available.append(1)
                        features.append(self.feat_map[feat])
                lnptext_given_t = self.ElnRho[features, :] + np.array(available)[:, None]
                lnptext_given_t -= logsumexp(lnptext_given_t, axis=1)[:, None]

            self.lnB = self.lnB[None, :, :] + lnptext_given_t[:, None, :]

            C_data = []
            for model in self.data_model:
                C_data.append(model.predict(doc_start, text))

            q_t = self._update_t(parallel, C, C_data, doc_start)

            seq = self._most_probable_sequence(C, C_data, doc_start, features, parallel)[1]

        return q_t, seq

    def annotator_accuracy(self):
        if self.alpha.ndim == 3:
            annotator_acc = self.alpha[np.arange(self.L), np.arange(self.L), :] \
                        / np.sum(self.alpha, axis=1)
        elif self.alpha.ndim == 2:
            annotator_acc = self.alpha[1, :] / np.sum(self.alpha[:2, :], axis=0)
        elif self.alpha.ndim == 4:
            annotator_acc = np.sum(self.alpha, axis=2)[np.arange(self.L), np.arange(self.L), :] \
                        / np.sum(self.alpha, axis=(1,2))

        if self.beta.ndim == 2:
            beta = np.sum(self.beta, axis=0)
        else:
            beta = self.beta

        annotator_acc *= (beta / np.sum(beta))[:, None]
        annotator_acc = np.sum(annotator_acc, axis=0)

        return annotator_acc

    def informativeness(self):

        ptj = np.zeros(self.L)
        for j in range(self.L):
            ptj[j] = np.sum(self.beta0[:, j]) + np.sum(self.q_t == j)

        entropy_prior = -np.sum(ptj * np.log(ptj))

        ptj_c = np.zeros((self.L, self.L, self.K))
        for j in range(self.L):
            if self.alpha.ndim == 4:
                ptj_c[j] = np.sum(self.alpha[j, :, :, :], axis=1) / np.sum(self.alpha[j, :, :, :], axis=(0,1))[None, :] * ptj[j]
            else:
                ptj_c[j] = self.alpha[j, :, :] / np.sum(self.alpha[j, :, :], axis=0)[None, :] * ptj[j]

        ptj_giv_c = ptj_c / np.sum(ptj_c, axis=0)[None, :, :]

        entropy_post = -np.sum(ptj_c * np.log(ptj_giv_c), axis=(0,1))

        return entropy_prior - entropy_post


    def _converged(self):
        '''
        Calculates whether the bsc has _converged or the maximum number of iterations is reached.
        The bsc has _converged when the maximum difference of an entry of q_t between two iterations is
        smaller than the given epsilon.
        '''
        if self.verbose:
            print("BSC: max. difference in posteriors at iteration %i: %.5f" % (self.iter, np.max(np.abs(self.q_t_old - self.q_t))))

        if not self.workers_converged:
            converged = self.iter >= self.max_iter
        else:
            converged = self.data_model_updated >= self.max_data_updates_at_end

        if not converged and self.data_model_updated != 1: # we need to allow at least one more iteration after the data
            # model has been updated because it will not yet have changed q_t
            converged = np.max(np.abs(self.q_t_old - self.q_t)) < self.eps

        if converged and not self.workers_converged:
            # the worker model has converged, but we need to do more iterations to converge the data model
            self.workers_converged = True
            converged = False

        return converged

def _log_dir(alpha, lnPi, sum_dim):
    x = (alpha - 1) * lnPi
    gammaln_alpha = gammaln(alpha)
    invalid_alphas = np.isinf(gammaln_alpha) | np.isinf(x) | np.isnan(x)
    gammaln_alpha[invalid_alphas] = 0  # these possibilities should be excluded
    x[invalid_alphas] = 0
    x = np.sum(x, axis=sum_dim)
    z = gammaln(np.sum(alpha, sum_dim)) - np.sum(gammaln_alpha, sum_dim)
    z[np.isinf(z)] = 0
    return np.sum(x + z)

# ----------------------------------------------------------------------------------------------------------------------


def _expec_t(lnR_, lnLambda):
    '''
    Calculate label probabilities for each token.
    '''
    return np.exp(lnR_ + lnLambda - logsumexp(lnR_ + lnLambda, axis=1)[:, None])


def _expec_joint_t(lnR_, lnLambda, lnB, C, C_data, doc_start, nscores, worker_model, blanks,
                   before_doc_idx=-1):
    '''
    Calculate joint label probabilities for each pair of tokens.
    '''
    # initialise variables
    L = lnB.shape[-1]
    K = C.shape[-1]

    lnS = np.repeat(lnLambda[:, None, :], L + 1, 1)

    mask = (C != 0)

    C = C - 1

    # flags to indicate whether the entries are valid or should be zeroed out later.
    flags = -np.inf * np.ones_like(lnS)
    flags[np.where(doc_start == 1)[0], before_doc_idx, :] = 0
    flags[np.where(doc_start == 0)[0], :L, :] = 0

    Cprev = np.append(np.zeros((1, K), dtype=int) + before_doc_idx, C[:-1, :], axis=0)
    Cprev[doc_start.flatten(), :] = before_doc_idx
    Cprev[Cprev == -1] = before_doc_idx

    for l in range(L):

        # Non-doc-starts
        lnPi_terms = worker_model.read_lnPi(l, C, Cprev, np.arange(K)[None, :], nscores, blanks)
        if C_data is not None and len(C_data):
            lnPi_data_terms = np.zeros(C_data[0].shape[0], dtype=float)

            for model_idx in range(len(C_data)):
                C_data_prev = np.append(np.zeros((1, L), dtype=int), C_data[model_idx][:-1, :], axis=0)
                C_data_prev[doc_start.flatten(), :] = 0
                C_data_prev[doc_start.flatten(), before_doc_idx] = 1
                C_data_prev[0, before_doc_idx] = 1

                for m in range(L):
                    weights = C_data[model_idx][:, m:m + 1] * C_data_prev

                    for n in range(L):
                        lnPi_data_terms_mn = weights[:, n] * worker_model.read_lnPi_data(l, m, n, 0, nscores, model_idx)
                        lnPi_data_terms_mn[np.isnan(lnPi_data_terms_mn)] = 0
                        lnPi_data_terms += lnPi_data_terms_mn
        else:
            lnPi_data_terms = 0

        loglikelihood_l = (np.sum(lnPi_terms * mask, 1) + lnPi_data_terms)[:, None]

        # for the other times. The document starts will get something invalid written here too, but it will be killed off by the flags
        # print('_expec_joint_t_quick: For class %i: adding the R terms from previous data points' % l)
        lnS[:, :, l] += loglikelihood_l + lnB[:, :, l]
        lnS[1:, :L, l] += lnR_[:-1, :]

    # print('_expec_joint_t_quick: Normalising...')

    # normalise and return
    lnS = lnS + flags
    if np.any(np.isnan(np.exp(lnS))):
        print('_expec_joint_t: nan value encountered (1) ')

    lnS = lnS - logsumexp(lnS, axis=(1, 2))[:, None, None]

    if np.any(np.isnan(np.exp(lnS))):
        print('_expec_joint_t: nan value encountered')

    return np.exp(lnS)


def _doc_forward_pass(C, C_data, lnB, nscores, worker_model, before_doc_idx=1):
    '''
    Perform the forward pass of the Forward-Backward bsc (for a single document).
    '''
    T = C.shape[0]  # infer number of tokens
    L = lnB.shape[-1]  # infer number of labels
    K = C.shape[-1]  # infer number of annotators
    Krange = np.arange(K)

    # initialise variables
    lnR_ = np.zeros((T, L))

    mask = C != 0

    # For the first data point
    lnPi_terms = worker_model.read_lnPi(None, C[0:1, :] - 1, before_doc_idx, Krange[None, :], nscores)
    lnPi_data_terms = 0

    if C_data is not None:
        for m in range(nscores):
            for model_idx in range(len(C_data)):
                lnPi_data_terms += (worker_model.read_lnPi_data(None, m, before_doc_idx, 0, nscores, model_idx)[:, :, 0]
                                    * C_data[model_idx][0, m])[:, 0]

    lnR_[0, :] = lnB[0, before_doc_idx, :] + np.dot(lnPi_terms[:, 0, :], mask[0, :][:, None])[:, 0] + lnPi_data_terms
    lnR_[0, :] = lnR_[0, :] - logsumexp(lnR_[0, :])

    Cprev = C - 1
    Cprev[Cprev == -1] = before_doc_idx
    Cprev = Cprev[:-1, :]
    Ccurr = C[1:, :] - 1

    # the other data points
    lnPi_terms = worker_model.read_lnPi(None, Ccurr, Cprev, Krange[None, :], nscores)
    lnPi_data_terms = 0
    if C_data is not None:
        for m in range(nscores):
            for n in range(nscores):
                for model_idx in range(len(C_data)):
                    lnPi_data_terms += worker_model.read_lnPi_data(None, m, n, 0, nscores, model_idx)[:, :, 0] * \
                                       C_data[model_idx][1:, m][None, :] * C_data[model_idx][:-1, n][None, :]

    likelihood_next = np.sum(mask[None, 1:, :] * lnPi_terms, axis=2) + lnPi_data_terms  # L x T-1
    # lnPi[:, Ccurr, Cprev, Krange[None, :]]

    # iterate through all tokens, starting at the beginning going forward
    for t in range(1, T):
        # iterate through all possible labels
        # prev_idx = Cprev[t - 1, :]
        lnR_t = logsumexp(lnR_[t - 1, :][:, None] + lnB[t, :, :], axis=0) + likelihood_next[:, t-1][None, :]
        # , np.sum(mask[t, :] * lnPi[:, C[t, :] - 1, prev_idx, Krange], axis=1)

        # normalise
        lnR_[t, :] = lnR_t - logsumexp(lnR_t)

    return lnR_


def _parallel_forward_pass(parallel, C, Cdata, lnB, doc_start, nscores, worker_model,
                           before_doc_idx=1):
    '''
    Perform the forward pass of the Forward-Backward bsc (for multiple documents in parallel).
    '''
    # split into documents
    C_by_doc = np.split(C, np.where(doc_start == 1)[0][1:], axis=0)
    lnB_by_doc = np.split(lnB, np.where(doc_start==1)[0][1:], axis=0)

    #print('Complete size of C = %s and lnB = %s' % (str(C.shape), str(lnB.shape)))

    C_data_by_doc = []

    # run forward pass for each doc concurrently
    # option backend='threading' does not work here because of the shared objects locking. Can we release them read-only?
    if Cdata is not None and len(Cdata) > 0:
        for m, Cdata_m in enumerate(Cdata):
            C_data_by_doc.append(np.split(Cdata_m, np.where(doc_start == 1)[0][1:], axis=0))
        if len(Cdata) >= 1:
            C_data_by_doc = list(zip(*C_data_by_doc))

        res = parallel(delayed(_doc_forward_pass)(C_doc, C_data_by_doc[d], lnB_by_doc[d],
                                                  nscores, worker_model, before_doc_idx)
                       for d, C_doc in enumerate(C_by_doc))
    else:
        res = parallel(delayed(_doc_forward_pass)(C_doc, None, lnB_by_doc[d],
                                                  nscores, worker_model, before_doc_idx)
                       for d, C_doc in enumerate(C_by_doc))

        # res = [_doc_forward_pass(C_doc, None, lnB_by_doc[d], lnPi, lnPi_data,
        #                                           nscores, worker_model, before_doc_idx)
        #                for d, C_doc in enumerate(C_by_doc)]

    # reformat results
    lnR_ = np.concatenate(res, axis=0)

    return lnR_


def _doc_backward_pass(C, C_data, lnB, nscores, worker_model, before_doc_idx=1):
    '''
    Perform the backward pass of the Forward-Backward bsc (for a single document).
    '''
    # infer number of tokens, labels, and annotators
    T = C.shape[0]
    L = lnB.shape[-1]
    K = C.shape[-1]
    Krange = np.arange(K)

    # initialise variables
    lnLambda = np.zeros((T, L))

    mask = (C != 0)

    Ccurr = C - 1
    Ccurr[Ccurr == -1] = before_doc_idx
    Ccurr = Ccurr[:-1, :]
    Cnext = C[1:, :] - 1

    lnPi_terms = worker_model.read_lnPi(None, Cnext, Ccurr, Krange[None, :], nscores)
    lnPi_data_terms = 0
    if C_data is not None:
        for model_idx in range(len(C_data)):

            C_data_curr = C_data[model_idx][:-1, :]
            C_data_next = C_data[model_idx][1:, :]

            for m in range(nscores):
                for n in range(nscores):
                    terms_mn = worker_model.read_lnPi_data(None, m, n, 0, nscores, model_idx)[:, :, 0] \
                               * C_data_next[:, m][None, :] \
                               * C_data_curr[:, n][None, :]
                    terms_mn[:, (C_data_next[:, m] * C_data_curr[:, n]) == 0] = 0
                    lnPi_data_terms += terms_mn

    likelihood_next = np.sum(mask[None, 1:, :] * lnPi_terms, axis=2) + lnPi_data_terms

    # iterate through all tokens, starting at the end going backwards
    for t in range(T - 2, -1, -1):
        # logsumexp over the L classes of the next timestep
        lnLambda_t = logsumexp(lnB[t+1, :, :] + lnLambda[t + 1, :][None, :] + likelihood_next[:, t][None, :], axis=1)

        # logsumexp over the L classes of the current timestep to normalise
        lnLambda[t] = lnLambda_t - logsumexp(lnLambda_t)

    if (np.any(np.isnan(lnLambda))):
        print('backward pass: nan value encountered at indexes: ')
        print(np.argwhere(np.isnan(lnLambda)))

    return lnLambda


def _parallel_backward_pass(parallel, C, C_data, lnB, doc_start, nscores, worker_model,
                            before_doc_idx=1):
    '''
    Perform the backward pass of the Forward-Backward bsc (for multiple documents in parallel).
    '''
    # split into documents
    docs = np.split(C, np.where(doc_start == 1)[0][1:], axis=0)
    lnB_by_doc = np.split(lnB, np.where(doc_start==1)[0][1:], axis=0)

    C_data_by_doc = []

    if C_data is not None:
        for m, Cdata_m in enumerate(C_data):
            C_data_by_doc.append(np.split(Cdata_m, np.where(doc_start == 1)[0][1:], axis=0))
        if len(C_data) >= 1:
            C_data_by_doc = list(zip(*C_data_by_doc))

    if C_data is not None and len(C_data) > 0:
        res = parallel(delayed(_doc_backward_pass)(doc, C_data_by_doc[d], lnB_by_doc[d], nscores, worker_model,
                                                   before_doc_idx) for d, doc in enumerate(docs))
    else:
        res = parallel(delayed(_doc_backward_pass)(doc, None, lnB_by_doc[d], nscores, worker_model,
                                                   before_doc_idx) for d, doc in enumerate(docs))

    # reformat results
    lnLambda = np.concatenate(res, axis=0)

    return lnLambda


def _doc_most_probable_sequence(C, C_data, lnEB, L, nscores, K, worker_model, before_doc_idx):
    lnV = np.zeros((C.shape[0], L))
    prev = np.zeros((C.shape[0], L), dtype=int)  # most likely previous states

    mask = C != 0

    t = 0
    for l in range(L):
        lnPi_terms = worker_model.read_lnPi(l, C[t, :] - 1, before_doc_idx, np.arange(K), nscores)

        lnPi_data_terms = 0
        if C_data is not None:
            for model_idx, C_data_m in enumerate(C_data):
                for m in range(L):
                    terms_startm = C_data_m[t, m] * worker_model.read_lnPi_data(l, m, before_doc_idx,
                                                                                 0, nscores, model_idx)
                    if C_data_m[t, m] == 0:
                        terms_startm = 0

                    lnPi_data_terms += terms_startm

        likelihood_current = np.sum(mask[t, :] * lnPi_terms) + lnPi_data_terms

        lnV[t, l] = lnEB[t, before_doc_idx, l] + likelihood_current


    Cnext = C[1:, :] - 1
    Cprev = C[:-1, :] - 1
    Cprev[Cprev == -1] = before_doc_idx

    lnPi_terms = worker_model.read_lnPi(None, Cnext, Cprev, np.arange(K), nscores)
    lnPi_data_terms = 0
    if C_data is not None:
        for model_idx, C_data_m in enumerate(C_data):
            for m in range(L):
                for n in range(L):
                    weights = (C_data_m[1:, m] * C_data_m[:-1, n])[None, :]
                    terms_mn = weights * worker_model.read_lnPi_data(None,
                                                 m, n, 0, nscores, model_idx)[:, :, 0]
                    terms_mn[:, weights[0] == 0] = 0

                    lnPi_data_terms += terms_mn

    likelihood_next = np.sum(mask[1:, :][None, :, :] * lnPi_terms, axis=2) + lnPi_data_terms

    for t in range(1, C.shape[0]):
        for l in range(L):
            p_current = lnV[t - 1, :] + lnEB[t, :L, l] + likelihood_next[l, t - 1]
            lnV[t, l] = np.max(p_current)
            prev[t, l] = np.argmax(p_current, axis=0)

        lnV[t, :] -= logsumexp(lnV[t, :]) # normalise to prevent underflow in long sequences

    # decode
    seq = np.zeros(C.shape[0], dtype=int)

    t = C.shape[0] - 1

    seq[t] = np.argmax(lnV[t, :])
    pseq = np.exp(np.max(lnV[t, :]))

    for t in range(C.shape[0] - 2, -1, -1):
        seq[t] = prev[t + 1, seq[t + 1]]

    return pseq, seq
