'''
Bayesian sequence combination, variational Bayes implementation.
'''
import datetime
import copy
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
from bsc.annotator_model import log_dirichlet_pdf

class BSC(object):

    K = None  # number of annotators
    L = None  # number of class labels

    lnB = None  # transition matrix
    lnPi = None  # worker confusion matrices

    Et = None  # current true label estimates
    Et_old = None  # previous true labels estimates

    iter = 0  # current iteration

    max_iter = None  # maximum number of iterations
    eps = None  # maximum difference of estimate differences in convergence chack

    def __init__(self, L=3, K=5, max_iter=20, eps=1e-4, inside_labels=[0], outside_label=1, beginning_labels=[2],
                 alpha0_diags=1.0, alpha0_factor=1.0, alpha0_B_factor=1.0, beta0_factor=1.0, nu0=1,
                 worker_model='ibcc', data_model=None, tagging_scheme='IOB2', transition_model='HMM', no_words=False,
                 model_dir=None, reload_lstm=False, embeddings_file=None, rare_transition_pseudocount=1e-12,
                 verbose=False, use_lowerbound=True):

        self.verbose = verbose

        self.use_lb = use_lowerbound
        if self.use_lb:
            self.lb = -np.inf

        # tagging_scheme can be 'IOB2' (all annos start with B) or 'IOB' (only annos
        # that follow another start with B).

        self.L = L
        self.K = K

        self.nu0 = nu0

        # choose whether to use the HMM transition model or not
        if transition_model == 'HMM':
            self.LM = MarkovLabelModel(np.ones((self.L, self.L)) * beta0_factor, self.L, verbose,
                                       tagging_scheme, beginning_labels, inside_labels,
                                       outside_label, rare_transition_pseudocount)
        else:
            self.LM = IndependentLabelModel(np.ones((self.L)) * beta0_factor, self.L, verbose)

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

        elif worker_model == 'mace' or worker_model == 'spam':
            self.A = MACEWorker(alpha0_diags, alpha0_factor, L, len(data_model))
            self.alpha_shape = (2 + self.L)

        elif worker_model == 'ibcc' or worker_model == 'CM':
            self.A = ConfusionMatrixWorker(alpha0_diags, alpha0_factor, L, len(data_model))
            self.alpha_shape = (self.L, self.L)

        elif worker_model == 'seq':
            self.A = SequentialWorker(alpha0_diags, alpha0_factor, L, len(data_model), tagging_scheme,
                                      beginning_labels, inside_labels, outside_label, alpha0_B_factor,
                                      rare_transition_pseudocount)
            self.alpha_shape = (self.L, self.L)

        elif worker_model == 'vec' or worker_model == 'CV':
            self.A = VectorWorker(alpha0_diags, alpha0_factor, L, len(data_model))
            self.alpha_shape = (self.L, self.L)

        self.alpha0_B_factor = alpha0_B_factor

        self.rare_transition_pseudocount = rare_transition_pseudocount

        self.outside_label = outside_label # identifies which true class value is assumed for the label before the start of a document

        self.max_iter = max_iter #- self.max_data_updates_at_end  # maximum number of iterations before training data models
        self.eps = eps  # threshold for convergence
        self.iter = 0
        self.max_internal_iters = 20

        self.verbose = False  # can change this if you want progress updates to be printed\


    def lowerbound(self):
        '''
        Compute the variational lower bound on the log marginal likelihood.

        TODO: it looks like this still has a small error in it, but this may be due to machine precision.
        '''
        lnq_Cdata = 0

        for midx, Cdata in enumerate(self.C_data):
            lnq_Cdata_m = Cdata * np.log(Cdata)
            lnq_Cdata_m[Cdata == 0] = 0
            lnq_Cdata += lnq_Cdata_m

        lnq_Cdata = np.sum(lnq_Cdata)

        lnpPi, lnqPi = self.A.lowerbound_terms()

        lnpCtB, lnqtB = self.LM.lowerbound_terms()

        if self.no_words:
            lnpRho = 0
            lnqRho = 0
        else:
            lnpwords = np.sum(self.ElnRho[self.features, :] * self.Et)
            nu0 = np.zeros((1, self.L)) + self.nu0
            lnpRho = np.sum(gammaln(np.sum(nu0)) - np.sum(gammaln(nu0)) + sum((nu0 - 1) * self.ElnRho, 0)) + lnpwords
            lnqRho = np.sum(gammaln(np.sum(self.nu,0)) - np.sum(gammaln(self.nu),0) + sum((self.nu - 1) * self.ElnRho, 0))

        print('Computing LB: %f, %f, %f' % (lnpCtB-lnq_Cdata-lnqtB,
                                                                lnpPi-lnqPi, lnpRho-lnqRho))

        lb = lnpCtB - lnq_Cdata + lnpPi - lnqPi - lnqtB + lnpRho - lnqRho
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

            self.A.alpha0 = np.exp(hyperparams[0:n_alpha_elements]).reshape(self.alpha_shape)
            self.LM.beta0 = np.ones((self.L, self.L)) * np.exp(hyperparams[-1])

            # run the method
            self.run(C, doc_start, features, converge_workers_first=converge_workers_first)

            # compute lower bound
            lb = self.lowerbound()

            print("Run %i. Lower bound: %.5f, alpha_0 = %s, nu0 scale = %.3f" % (self.opt_runs,
                                                 lb, str(np.exp(hyperparams[0:-1])), np.exp(hyperparams[-1])))

            self.opt_runs += 1
            return -lb

        initialguess = np.log(np.append(self.A.alpha0.flatten(), self.LM.beta0[0, 0]))
        ftol = 1.0  #1e-3
        opt_hyperparams, _, _, _, _ = fmin(neg_marginal_likelihood, initialguess, args=(C, doc_start), maxfun=maxfun,
                                                     full_output=True, ftol=ftol, xtol=1e100)

        print("Optimal hyper-parameters: alpha_0 = %s, nu0 scale = %s" % (np.array2string(np.exp(opt_hyperparams[:-1])),
                                                                          np.array2string(np.exp(opt_hyperparams[-1]))))

        return self.Et, self.LM.most_probable_sequence(self.word_ll(self.features))[1]


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

        self.nu = self.nu0 + self.features_mat.dot(self.Et) # nfeatures x nclasses

        # update the expected log word likelihoods
        self.ElnRho = psi(self.nu) - psi(np.sum(self.nu, 0)[None, :])

        # get the expected log word likelihoods of each token
        lnptext_given_t = self.ElnRho[self.features, :]
        lnptext_given_t -= logsumexp(lnptext_given_t, axis=1)[:, None]

        return lnptext_given_t # N x nclasses where N is number of tokens/data points


    def word_ll(self, features):
        if self.no_words:
            word_ll = None
        else:
            ERho = self.nu / np.sum(self.nu, 0)[None, :]  # nfeatures x nclasses
            lnERho = np.zeros_like(ERho)
            lnERho[ERho != 0] = np.log(ERho[ERho != 0])
            lnERho[ERho == 0] = -np.inf
            word_ll = lnERho[features, :]

        return word_ll


    def run(self, C, doc_start, features=None, converge_workers_first=True, crf_probs=False, dev_sentences=[],
            gold_labels=None, C_data_initial=None):
        '''
        Runs the BAC bsc with the given annos and list of document starts.

        C: use -1 to indicate where the annos are missing. In future we may want to change
        this to a sparse format?

        '''
        if self.verbose:
            print('BSC: run() called with C shape = %s' % str(C.shape))

        # if there are any gold labels, supply a vector or list of values with -1 for the missing ones
        if gold_labels is not None:
            self.gold = np.array(gold_labels).flatten().astype(int)
        else:
            self.gold = None

        # initialise the hyperparameters to correct sizes
        self.A._expand_alpha0(self.K, self.L)

        # validate input data
        assert C.shape[0] == doc_start.shape[0]

        if self.verbose:
            print('Aggregating annotation matrix with shape = %s' % str(C.shape))

        # C = C.astype(int) + 1
        doc_start = doc_start.astype(bool)
        if doc_start.ndim == 1:
            doc_start = doc_start[:, None]

        self.N = doc_start.size

        if self.iter == 0:
            # initialise word model and worker models
            self._init_words(features)
            self.A._init_lnPi()

            # initialise variables
            self.Et_old = np.zeros((C.shape[0], self.L))
            self.Et = np.ones((C.shape[0], self.L))

            #oldlb = -np.inf

            self.doc_start = doc_start
            self.C = C.astype(int)

            self.blanks = C == -1

        self.C_data = []
        for midx, model in enumerate(self.data_model):
            model.init(self.C.shape[0], features, doc_start, self.L,
                      self.max_data_updates_at_end if converge_workers_first else self.max_iter, crf_probs,
                      dev_sentences, self.A, self.max_internal_iters)
            self.A.q_pi_data(midx)

            if C_data_initial is None:
                self.C_data.append(np.zeros((self.C.shape[0], self.L)) ) #+ (1.0 / self.L)
            else:
                self.C_data.append(C_data_initial[midx])
                print('integrated model initial labels: ' +
                      str(np.unique(np.argmax(C_data_initial[midx], axis=1))))

        print('Parallel can run %i jobs simultaneously, with %i cores' % (effective_n_jobs(), cpu_count()) )

        self.data_model_updates = 0
        self.workers_converged = False
        self.converge_workers_first = converge_workers_first

        # timestamp = datetime.datetime.now().strftime('started-%Y-%m-%d-%H-%M-%S')

        # main inference loop
        with Parallel(n_jobs=-1, backend='threading') as parallel: #) as parallel:

            while not self._converged() or not self.workers_converged:

                # print status if desired
                if self.verbose:
                    print("BAC iteration %i in progress" % self.iter)

                self.Et_old = self.Et

                if self.iter > 0:
                    self.Et = self.LM.update_t(parallel, self.C_data)
                else:
                    self.Et = self.LM.init_t(self.C, doc_start, self.A, self.blanks, self.gold)

                if self.verbose:
                    print("BAC iteration %i: computed label sequence probabilities" % self.iter)

                # update E_lnB
                ln_word_likelihood = self._update_words()
                self.LM.update_B(ln_word_likelihood)
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
                            self.C_data[midx] = model.fit_predict(self.Et)
                        elif model.train_type == 'MLE':
                            seq = self.LM.most_probable_sequence(self.word_ll(self.features))[1]
                            self.C_data[midx] = model.fit_predict(seq)

                        if self.verbose:
                            print("BAC iteration %i: updated feature-based predictions from %s" %
                                  (self.iter, type(model)))


                        print('integrated model predicted the following labels: ' +
                              str(np.unique(np.argmax(self.C_data[midx], axis=1))))

                    if not converge_workers_first or self.workers_converged or C_data_initial is not None:

                        self.A.update_post_alpha_data(midx, self.Et, self.C_data[midx], doc_start, self.L)
                        self.A.q_pi_data(midx)

                        if self.verbose:
                            print("BAC iteration %i: updated model for feature-based predictor of type %s" %
                                  (self.iter, str(type(model))) )

                        # if type(model) == LSTM:
                        #    np.save('LSTM_worker_model_%s.npy' % timestamp, model.alpha_data)
                    elif C_data_initial is None: # model is switched off
                        self.C_data[midx][:] = 0

                if not converge_workers_first or self.workers_converged:
                    self.data_model_updates += 1 # count how many times we updated the data models

                # update E_lnpi
                self.A.update_post_alpha(self.Et, self.C, doc_start, self.L)
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

            pseq, seq = self.LM.most_probable_sequence(self.word_ll(self.features))
        if self.verbose:
            print("BAC iteration %i: fitting/predicting complete." % self.iter)

            # print('BAC final transition matrix: ')
            # print(np.around(self.beta / (np.sum(self.beta, -1)[:, None] if self.beta.ndim > 1 else np.sum(self.beta)), 2))

        return self.Et, seq, pseq


    def predict(self, doc_start, text):

        self.LM.C_by_doc = None

        with Parallel(n_jobs=-1, backend='threading') as parallel:

            doc_start = doc_start.astype(bool)
            self.C = np.zeros((len(doc_start), self.K), dtype=int) - 1  # all blank
            self.blanks = self.C == 0

            self.LM.lnB = psi(self.LM.beta) - psi(np.sum(self.LM.beta, -1))[:, None]

            if self.no_words:
                lnptext_given_t = np.zeros((len(doc_start), self.L))
                self.features = None
            else:
                # get the expected log word likelihoods of each token
                self.features = []  # list of mapped index values for each token
                available = []
                for feat in text.flatten():
                    if feat not in self.feat_map:
                        available.append(0)
                        self.features.append(0)
                    else:
                        available.append(1)
                        self.features.append(self.feat_map[feat])
                lnptext_given_t = self.ElnRho[self.features, :] + np.array(available)[:, None]
                lnptext_given_t -= logsumexp(lnptext_given_t, axis=1)[:, None]

            self.LM.lnB = self.LM.lnB[None, :, :] + lnptext_given_t[:, None, :]

            C_data = []
            for model in self.data_model:
                C_data.append(model.predict(doc_start, text))

            self.LM.init_t(self.C, doc_start, self.A, self.blanks)
            q_t = self.LM.update_t(parallel, C_data)

            seq = self.LM.most_probable_sequence(self.word_ll(self.features))[1]

        return q_t, seq


    def _convergence_diff(self):
        if self.use_lb:

            oldlb = self.lb
            self.lb = self.lowerbound()

            return self.lb - oldlb
        else:
            return np.max(np.abs(self.Et_old - self.Et))


    def _converged(self):
        '''
        Calculates whether the bsc has _converged or the maximum number of iterations is reached.
        The bsc has _converged when the maximum difference of an entry of q_t between two iterations is
        smaller than the given epsilon.
        '''
        if self.iter <= 1:
            return False # don't bother to check because some of the variables have not even been initialised yet!


        if self.workers_converged:
            # do we have to wait for data models to converge now?
            converged = (not self.converge_workers_first) or (self.data_model_updates >= self.max_data_updates_at_end)
        else:
            # have we run out of iterations for learning worker models?
            converged = self.iter >= self.max_iter

        # We have not reached max number of iterations yet. Check to see if updates have already converged.
        # Don't perform this check if the data_model has been updated once: it will not yet have resulted in changes to
        # self.Et, so diff will be ~0 but we need to do at least one more iteration.
        # If self.data_model_updates == 0 then data model is either not used or not yet updated, in which case, check
        # for convergence of worker models now.
        if not converged and self.data_model_updates != 1:

            diff = self._convergence_diff()

            if self.verbose:
                print("BSC: max. difference at iteration %i: %.5f" % (self.iter, diff))

            converged = diff < self.eps

        # The worker models have converged now, but we may still have data models to update
        if converged and not self.workers_converged:
            self.workers_converged = True
            if self.converge_workers_first:
                converged = False

        return converged

# ----------------------------------------------------------------------------------------------------------------------

class LabelModel(object):
    def __init__(self, beta0, L, outside_label, verbose=False):
        self.L = L
        self.outside_label = outside_label
        self.verbose = verbose

        self.beta0 = beta0

    def update_B(self, ln_word_likelihood):
        pass

    def init_t(self, C, doc_start, A, blanks, gold=None):

        self.C = C
        self.ds = doc_start.flatten() == 1
        self.A = A
        self.N = C.shape[0]
        self.blanks = blanks
        self.gold = gold

        self.beta = self.beta0

        if self.beta0.ndim >= 2:
            beta0_sum = psi(np.sum(self.beta0, -1))[:, None]
        else:
            beta0_sum = psi(np.sum(self.beta0))

        self.lnB = psi(self.beta0) - beta0_sum
        self.lnB = np.tile(self.lnB, (self.N, 1, 1))

        self.Cprev = np.append(np.zeros((1, self.C.shape[1]), dtype=int) - 1, self.C[:-1, :], axis=0)


    def update_t(self):
        pass

    def most_probable_sequence(self, word_ll):
        pass

    def lowerbound_terms(self):
        pass

class MarkovLabelModel(LabelModel):
    def __init__(self, beta0, L, verbose, tagging_scheme,
                 beginning_labels, inside_labels, outside_label, rare_transition_pseudocount):

        self.C_by_doc = None
        self.lnR_ = None

        super().__init__(beta0, L, outside_label, verbose)

        if self.beta0.ndim != 2:
            return

        # set priors for invalid transitions (to low values)
        if tagging_scheme == 'IOB':
            restricted_labels = beginning_labels  # labels that cannot follow an outside label
            unrestricted_labels = inside_labels  # labels that can follow any label
        elif tagging_scheme == 'IOB2':
            restricted_labels = inside_labels
            unrestricted_labels = beginning_labels

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


    def update_B(self, ln_word_likelihood):
        '''
        Update the transition model.
        '''
        self.beta = self.beta0 + np.sum(self.Et_joint, 0)
        self.lnB = psi(self.beta) - psi(np.sum(self.beta, -1))[:, None]

        self.lnB = self.lnB[None, :, :] + ln_word_likelihood[:, None, :]

        if np.any(np.isnan(self.lnB)):
            print('update_B: nan value encountered!')

    def init_t(self, C, doc_start, A, blanks, gold=None):

        super().init_t(C, doc_start, A, blanks, gold)

        doc_start = doc_start.flatten()

        B = self.beta0 / np.sum(self.beta0, axis=1)[:, None]

        # initialise using Dawid and Skene
        self.Et = ds(C, self.L, 0.1)

        self.Et_joint = np.zeros((self.N, self.L, self.L))
        for l in range(self.L):
            self.Et_joint[1:, l, :] = self.Et[:-1, l][:, None] * self.Et[1:, :] * B[l, :][None, :]

        self.Et_joint[doc_start, :, :] = 0
        self.Et_joint[doc_start, self.outside_label, :] = self.Et[doc_start, :] * B[self.outside_label, :]

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

        # calculate alphas and betas using forward-backward bsc
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
        Perform the forward pass of the Forward-Backward bsc (for multiple documents in parallel).
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

            res = self.parallel(delayed(_doc_forward_pass)(C_doc, self.C_data_by_doc[d], self.blanks_by_doc[d],
                                                           self.lnB_by_doc[d],
                                                           self.A, self.outside_label)
                           for d, C_doc in enumerate(self.C_by_doc))
        else:
            res = self.parallel(delayed(_doc_forward_pass)(C_doc, None, self.blanks_by_doc[d], self.lnB_by_doc[d],
                                                           self.A, self.outside_label)
                           for d, C_doc in enumerate(self.C_by_doc))

        # reformat results
        self.lnR_ = np.concatenate([r[0] for r in res], axis=0)
        self.scaling = [r[1] for r in res]


    def _parallel_backward_pass(self):
        '''
        Perform the backward pass of the Forward-Backward bsc (for multiple documents in parallel).
        '''
        if self.Cdata is not None and len(self.Cdata) > 0:
            res = self.parallel(delayed(_doc_backward_pass)(doc, self.C_data_by_doc[d], self.blanks_by_doc[d],
                                                            self.lnB_by_doc[d], self.L,
                                                            self.A, self.scaling[d]
                                                    ) for d, doc in enumerate(self.C_by_doc))
        else:
            res = self.parallel(delayed(_doc_backward_pass)(doc, None, self.blanks_by_doc[d], self.lnB_by_doc[d],
                                                            self.L, self.A, self.scaling[d],
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
            lnPi_terms = self.A.read_lnPi(l, self.C, self.Cprev, np.arange(K)[None, :], self.L, self.blanks)
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
                            lnPi_data_terms_mn = weights[:, n] * self.A.read_lnPi_data(l, m, n, self.L, model_idx)
                            lnPi_data_terms_mn[np.isnan(lnPi_data_terms_mn)] = 0
                            lnPi_data_terms += lnPi_data_terms_mn
            else:
                lnPi_data_terms = 0

            loglikelihood_l = (np.sum(lnPi_terms, 1) + lnPi_data_terms)[:, None]

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


    def most_probable_sequence(self, word_ll):
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

            res = self.parallel(delayed(_doc_most_probable_sequence)(doc, C_data_by_doc[d], lnEB_by_doc[d], self.L,
                                                         self.C.shape[1], self.A, self.outside_label)
                           for d, doc in enumerate(self.C_by_doc))
        else:
            res = self.parallel(delayed(_doc_most_probable_sequence)(doc, None, lnEB_by_doc[d], self.L,
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
        qt_sum[qt_sum==0] = 1.0 # doesn't matter, they will be multiplied by zero. This avoids the warning
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


class IndependentLabelModel(LabelModel):

    def update_B(self, ln_word_likelihood):
        '''
        Update the transition model.

        q_t_joint is N x L x L. In this case, we just sum up over all previous label values, so the counts are
        independent of the previous labels.
        '''
        self.beta = self.beta0 + np.sum(self.Et, 0)

        self.lnB = psi(self.beta) - psi(np.sum(self.beta, -1))

        self.lnB = self.lnB[None, :] + ln_word_likelihood

        if np.any(np.isnan(self.lnB)):
            print('_update_B: nan value encountered!')

    def init_t(self, C, doc_start, A, blanks, gold=None):

        super().init_t(C, doc_start, A, blanks, gold)

        # initialise using the fraction of votes for each label
        self.Et = np.zeros((C.shape[0], self.L)) + self.beta0
        for l in range(self.L):
            self.Et[:, l] += np.sum(C == l, axis=1)
        self.Et /= np.sum(self.Et, axis=1)[:, None]

        return self.Et


    def update_t(self, parallel, C_data):
        Krange = np.arange(self.C.shape[1], dtype=int)

        self.lnp_Ct = np.copy(self.lnB)

        lnPi_data_terms = 0
        for model_idx in range(len(C_data)):

            C_m = C_data[model_idx]
            C_m_prev = np.concatenate((np.zeros((1, self.L)), C_data[model_idx][:-1, :]), axis=0)

            for m in range(self.L):
                for n in range(self.L):
                    lnPi_data_terms += self.A.read_lnPi_data(None, m, n, self.L, model_idx)[:, :, 0] * \
                                       C_m[:, m][None, :] * C_m_prev[:, n][None, :]

        Elnpi = self.A.read_lnPi(None, self.C, self.Cprev, Krange[None, :], self.L, self.blanks)
        self.lnp_Ct += (np.sum(Elnpi, axis=2) + lnPi_data_terms).T

        p_Ct = np.exp(self.lnp_Ct)
        self.Et = p_Ct / np.sum(p_Ct, axis=1)[:, None]

        if self.gold is not None:
            goldidxs = self.gold != -1
            self.Et[goldidxs, :] = 0
            self.Et[goldidxs, self.gold[goldidxs]] = 1.0

        return self.Et

    def most_probable_sequence(self, word_ll):
        return np.max(self.Et, axis=1), np.argmax(self.Et, axis=1)


    def lowerbound_terms(self):
        lnB = psi(self.beta) - psi(np.sum(self.beta, -1))
        lnB[np.isinf(lnB) | np.isnan(lnB)] = 0

        lnpt = self.lnp_Ct * self.Et
        lnpt = np.sum(lnpt)

        lnqt = np.log(self.Et)
        lnqt[self.Et == 0] = 0
        lnqt *= self.Et
        lnqt = np.sum(lnqt)

        lnpB = log_dirichlet_pdf(self.beta0, lnB, 0)
        lnqB = log_dirichlet_pdf(self.beta, lnB, 0)

        return lnpt + lnpB, lnqt + lnqB


def _doc_forward_pass(C, C_data, blanks, lnB, worker_model, outside_label):
    '''
    Perform the forward pass of the Forward-Backward bsc (for a single document).
    '''
    T = C.shape[0]  # infer number of tokens
    L = lnB.shape[-1]  # infer number of labels
    K = C.shape[-1]  # infer number of annotators
    Krange = np.arange(K)

    # initialise variables
    lnR_ = np.zeros((T, L))
    scaling = np.zeros(T)
    # blanks = C == -1
    # For the first data point
    lnPi_terms = worker_model.read_lnPi(None, C[0:1, :], np.zeros((1, K), dtype=int)-1, Krange[None, :], L, blanks[0:1])
    lnPi_data_terms = 0

    if C_data is not None:
        for m in range(L):
            for model_idx in range(len(C_data)):
                lnPi_data_terms += (worker_model.read_lnPi_data(None, m, outside_label, L, model_idx)[:, :, 0]
                                    * C_data[model_idx][0, m])[:, 0]

    lnR_[0, :] = lnB[0, outside_label, :] + np.sum(lnPi_terms[:, 0, :], axis=1) + lnPi_data_terms
    scaling[0] = logsumexp(lnR_[0, :])
    lnR_[0, :] = lnR_[0, :] - scaling[0]

    Cprev = C[:-1, :]
    Ccurr = C[1:, :]

    # the other data points
    lnPi_terms = worker_model.read_lnPi(None, Ccurr, Cprev, Krange[None, :], L, blanks[1:])
    lnPi_data_terms = 0
    if C_data is not None:
        for m in range(L):
            for n in range(L):
                for model_idx in range(len(C_data)):
                    lnPi_data_terms += worker_model.read_lnPi_data(None, m, n, L, model_idx)[:, :, 0] * \
                                       C_data[model_idx][1:, m][None, :] * C_data[model_idx][:-1, n][None, :]

    likelihood_next = np.sum(lnPi_terms, axis=2) + lnPi_data_terms  # L x T-1
    # lnPi[:, Ccurr, Cprev, Krange[None, :]]

    # iterate through all tokens, starting at the beginning going forward
    sum_trans_likelihood = lnB[1:] + likelihood_next.T[:, None, :]

    for t in range(1, T):
        # iterate through all possible labels
        # prev_idx = Cprev[t - 1, :]
        lnR_t = logsumexp(lnR_[t - 1, :][:, None] + sum_trans_likelihood[t-1], axis=0)
        # , np.sum(mask[t, :] * lnPi[:, C[t, :] - 1, prev_idx, Krange], axis=1)

        # normalise
        scaling[t] = logsumexp(lnR_t)
        lnR_[t, :] = lnR_t - scaling[t]

    return lnR_, scaling

def _doc_backward_pass(C, C_data, blanks, lnB, L, worker_model, scaling):
    '''
    Perform the backward pass of the Forward-Backward bsc (for a single document).
    '''
    # infer number of tokens, labels, and annotators
    T = C.shape[0]
    K = C.shape[-1]
    Krange = np.arange(K)

    # initialise variables
    lnLambda = np.zeros((T, L))

    Ccurr = C[:-1, :]
    Cnext = C[1:, :]
    blanks = blanks[1:]#Cnext == -1

    lnPi_terms = worker_model.read_lnPi(None, Cnext, Ccurr, Krange[None, :], L, blanks)
    lnPi_data_terms = 0
    if C_data is not None:
        for model_idx in range(len(C_data)):

            C_data_curr = C_data[model_idx][:-1, :]
            C_data_next = C_data[model_idx][1:, :]

            for m in range(L):
                for n in range(L):
                    terms_mn = worker_model.read_lnPi_data(None, m, n, L, model_idx)[:, :, 0] \
                               * C_data_next[:, m][None, :] \
                               * C_data_curr[:, n][None, :]
                    terms_mn[:, (C_data_next[:, m] * C_data_curr[:, n]) == 0] = 0
                    lnPi_data_terms += terms_mn

    likelihood_next = np.sum(lnPi_terms, axis=2) + lnPi_data_terms

    # iterate through all tokens, starting at the end going backwards
    for t in range(T - 2, -1, -1):
        # logsumexp over the L classes of the next timestep
        lnLambda_t = logsumexp(lnLambda[t + 1, :][None, :] + lnB[t+1, :, :] + likelihood_next[:, t][None, :], axis=1)

        # logsumexp over the L classes of the current timestep to normalise
        lnLambda[t] = lnLambda_t - scaling[t+1] # logsumexp(lnLambda_t)

    if (np.any(np.isnan(lnLambda))):
        print('backward pass: nan value encountered at indexes: ')
        print(np.argwhere(np.isnan(lnLambda)))

    return lnLambda

def _doc_most_probable_sequence(C, C_data, lnEB, L, K, worker_model, before_doc_idx):
    lnV = np.zeros((C.shape[0], L))
    prev = np.zeros((C.shape[0], L), dtype=int)  # most likely previous states

    for l in range(L):
        if l != before_doc_idx:
            lnEB[0, l, :] = -np.inf

    blanks = C == -1
    Cprev = np.concatenate((np.zeros((1, C.shape[1]), dtype=int) - 1, C[:-1, :]), axis=0)

    lnPi_terms = worker_model.read_lnPi(None, C, Cprev, np.arange(K), L, blanks)
    lnPi_data_terms = 0
    if C_data is not None:
        for model_idx, C_data_m in enumerate(C_data):
            C_data_m_prev = np.concatenate((np.zeros((1, L), dtype=int) - 1, C_data_m[:-1, :]), axis=0)
            for m in range(L):
                for n in range(L):
                    weights = (C_data_m[:, m] * C_data_m_prev[:, n])[None, :]
                    terms_mn = weights * worker_model.read_lnPi_data(None, m, n, L, model_idx)[:, :, 0]
                    terms_mn[:, weights[0] == 0] = 0

                    lnPi_data_terms += terms_mn

    likelihood = np.sum(lnPi_terms, axis=2) + lnPi_data_terms

    for t in range(0, C.shape[0]):
        for l in range(L):
            p_current = lnV[t - 1, :] + lnEB[t, :, l] + likelihood[l, t]
            lnV[t, l] = np.max(p_current)
            prev[t, l] = np.argmax(p_current, axis=0)

        lnV[t, :] -= logsumexp(lnV[t, :])  # normalise to prevent underflow in long sequences

    # decode
    seq = np.zeros(C.shape[0], dtype=int)

    t = C.shape[0] - 1

    seq[t] = np.argmax(lnV[t, :])
    pseq = np.exp(np.max(lnV[t, :]))

    for t in range(C.shape[0] - 2, -1, -1):
        seq[t] = prev[t + 1, seq[t + 1]]

    return pseq, seq
