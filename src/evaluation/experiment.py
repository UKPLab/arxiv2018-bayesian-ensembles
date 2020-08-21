'''
Created on Nov 1, 2016

@author: Melvin Laux and Edwin Simpson
'''

import logging
import pickle
import pandas as pd
import datetime
import os, subprocess
import sklearn.metrics as skm
import numpy as np

import taggers.lample_lstm_tagger.lstm_wrapper as lstm_wrapper
from baselines.dawid_and_skene import ds, ibccvb
from bayesian_combination.ibcc import IBCC, HeatmapBCC
from evaluation.metrics import calculate_scores
from evaluation.plots import SCORE_NAMES
from baselines.hmmcrowd import HMM_crowd
from baselines.util import crowd_data, data_to_hmm_crowd_format, subset_hmm_crowd_data
from baselines import ibcc, clustering, majority_voting
from bayesian_combination import bayesian_combination

logging.basicConfig(level=logging.DEBUG)


data_root_dir = '.'  # '../../../data/bayesian_sequence_combination/'
output_root_dir = os.path.join(data_root_dir, 'output')


def _append_to_csv(outputdir, method, method_idx, new_data, filename, file_identifier):

    filename = os.path.join(outputdir, '%s_%s.csv' % (filename, file_identifier))

    new_data = pd.DataFrame(new_data[:, method_idx], columns=[str(method).strip('[]')])

    if os.path.isfile(filename):
        data = pd.read_csv(filename, delimiter=',')
        expanded_data = pd.concat((data, new_data), axis=1)
    else:
        expanded_data = new_data

    expanded_data.to_csv(filename, index=False)


class Experiment(object):

    def __init__(self, outputdir, nclasses, annotations, gold, doc_start, text,
                 annos_val=None, gold_val=None, doc_start_val=None, text_val=None,
                 gold_nocrowd=None, doc_start_nocrowd=None, text_nocrowd=None,
                 alpha0_factor=1.0, alpha0_diags=1.0, beta0_factor=0.1, begin_factor=1.0,
                 max_iter=20, crf_probs=False, rep=0, bootstrapping=False
                 ):
        '''
        :param outputdir: the directory where results, predictions and any model files will be stored
        :param annotations: rows correspond to tokens, columns to annotators.
        :param gold: class labels for computing the performance metrics. Missing values should be set to -1.
        :param doc_start: binary vector indicating the start of each sequence/document/sentence.
        :param text: strings associated with each token.
        :param annos_val: crowdsourced labels for a validation set.
        :param gold_val: for tuning hyperparameters
        :param doc_start_val:
        :param text_val:
        :param gold_nocrowd: for evaluating on a second set of data points where crowd labels are not available
        :param doc_start_nocrowd:
        :param text_nocrowd: the features for testing the model trained on crowdsourced data to classify data with no labels at all
        :param nclasses: number of classes
        :param alpha0_factor: smoothing hyperparameter for annotator models.
        :param alpha0_diags: correctness bias hyperparameter for annotator models.
        :param beta0_factor: smoothing hyperparameter for prior over class labels.
        :param begin_factor: additional correctness bias for B tokens.
        :param max_iter: maximum iterations for iterative aggregation methods.
        :param crf_probs: LSTM method produces probabilities using a CRF output layer.
        :param rep: repetition number for an experiment that is carried out multiple times; sets a different random seed for each repetition.
        :param bootstrapping: calculate performance metrics using bootstrapping for small datasets
        '''

        self.methods = None
        self.num_classes = nclasses
        self.postprocess = False  # previous papers did not use this so we leave out to make results comparable.
        self.random_sampling = False

        if not os.path.exists(output_root_dir):
            os.mkdir(output_root_dir)

        self.outputdir = outputdir
        if not os.path.exists(outputdir):
                os.mkdir(outputdir)
        outputdir += '/'

        # Data -------------------------------------------------
        self.annos_test = annotations
        self.gold_test = gold
        self.doc_start_test = doc_start
        self.text_test = text

        self.annos_val = annos_val
        self.gold_val = gold_val
        self.doc_start_val = doc_start_val
        self.text_val = text_val

        self.gold_nocrowd = gold_nocrowd
        self.doc_start_nocrowd = doc_start_nocrowd
        self.text_nocrowd = text_nocrowd

        # Method hyperparameters and settings ---------------------------------
        self.crf_probs = crf_probs
        self.opt_hyper = False
        self.use_lb = False

        self.alpha0_factor = alpha0_factor
        self.alpha0_diags = alpha0_diags
        self.begin_factor = begin_factor
        self.beta0_factor = beta0_factor
        self.max_internal_iter = 3

        self.max_iter = max_iter # allow all methods to use a maximum no. iterations

        np.random.seed(3849)
        self.seed = np.random.randint(1, 1000, 100)[rep] # seeds for AL

        # Results -------------------------------------------------------
        # save results from methods here. If we use compound methods, we can reuse these results in different
        # combinations of methods.
        self.aggs = {}
        self.probs = {}

        self.bootstrapping = bootstrapping

        self.scores = None
        self.scores_nocrowd = None


    def tune_alpha0(self, alpha0diag_proposals, alpha0factor_proposals, beta0factor_proposals,
                    method, metric_idx_to_optimise=8, new_data=False):

        self.methods = [method]

        scores = np.zeros((len(beta0factor_proposals) * len(alpha0diag_proposals), len(alpha0factor_proposals)))

        best_scores = np.zeros(4) - np.inf
        best_idxs = np.zeros(4)

        for h, beta0factor in enumerate(beta0factor_proposals):
            self.beta0_factor = beta0factor

            for i, alpha0diag in enumerate(alpha0diag_proposals):
                self.alpha0_diags = alpha0diag

                for j, alpha0factor in enumerate(alpha0factor_proposals):
                    self.alpha0_factor = alpha0factor

                    # reset saved data so that models are run again.
                    self.aggs = {}
                    self.probs = {}
                    outputdir_ij = self.outputdir + ('_%i_%i_%i_' % (h, i, j)) + method + '/'

                    all_scores, _, _, _, _, _ = self.run_methods(outputdir_ij, new_data=new_data)

                    if metric_idx_to_optimise != 5: # maximise these scores
                        scores[(h*len(alpha0diag_proposals)) + i, j] = all_scores[metric_idx_to_optimise, :]
                    else: # minimise this score
                        scores[(h*len(alpha0diag_proposals)) + i, j] = -all_scores[metric_idx_to_optimise, :]

                    print('Scores for %f, %f, %f: %f' % (beta0factor, alpha0diag, alpha0factor,
                                                         scores[(h*len(alpha0diag_proposals)) + i, j]))

                    if scores[(h*len(alpha0diag_proposals)) + i, j] > best_scores[0]:
                        best_scores[0] = scores[(h*len(alpha0diag_proposals)) + i, j]
                        best_scores[1] = beta0factor
                        best_scores[2] = alpha0diag
                        best_scores[3] = alpha0factor

                        best_idxs[0] = scores[(h*len(alpha0diag_proposals)) + i, j]
                        best_idxs[1] = h
                        best_idxs[2] = i
                        best_idxs[3] = j

                    print('Saving scores for this setting to %s' % (self.outputdir + '/%s_scores.csv' % method))
                    np.savetxt(self.outputdir + '/%s_scores.csv' % method, scores, fmt='%s', delimiter=',',
                               header=str(self.methods).strip('[]'))

                    np.savetxt(self.outputdir + '/%s_bestscores.csv' % method, best_scores, fmt='%s', delimiter=',',
                               header=str(self.methods).strip('[]'))

        self.aggs = {}
        self.probs = {}
        return best_idxs


    def run_methods(self, test_on_dev=False, new_data=False,
                    active_learning=False, AL_batch_fraction=0.05, max_AL_iters=10,
                    save_with_timestamp=True):
        '''
        Run the aggregation methods and evaluate them.
        :param test_on_dev: use test rather than dev set
        :param new_data: set to true if all cached data files should be overwritten
        :param active_learning: set to true to run an AL simulation
        :param AL_batch_fraction: what proportion of labels are sampled at each AL iteration
        :param max_AL_iters: maximum number of AL rounds.
        :return:
        '''
        if active_learning:
            print('Running active learning on the test dataset.')

        if not test_on_dev:
            self.annos_all = self.annos_test
            self.annos = self.annos_test
            self.doc_start_all = self.doc_start_test
            self.doc_start = self.doc_start_test
            self.text_all = self.text_test
            self.text = self.text_test
            self.gold = self.gold_test
        else:
            self.annos_all = self.annos_val
            self.annos = self.annos_val
            self.doc_start_all = self.doc_start_val
            self.doc_start = self.doc_start_val
            self.text_all = self.text_val
            self.text = self.text_val
            self.gold = self.gold_val

        # a second test set with no crowd labels was supplied directly
        if self.gold_nocrowd is not None and self.text_nocrowd is not None and self.doc_start_nocrowd is not None:
            self.N_nocrowd = self.gold_nocrowd.shape[0]
        else:
            self.N_nocrowd = 0

        print('Running experiment on %s set%s' % ('dev' if test_on_dev else 'test', '.' if self.N_nocrowd==0 else
                ' and predicting on additional test data with no crowd labels'))

        Ntoks = self.annos.shape[0]
        Ndocs = np.sum(self.doc_start)
        # get total annotation count
        Nannos = np.sum(self.annos_all[(self.doc_start == 1).flatten(), :] != -1)
        print('Data set: %i documents, %i tokens, %i documents without crowd labels.' % (Ndocs, Ntoks, self.N_nocrowd))

        preds = -np.ones((Ntoks, len(self.methods)))
        probs_allmethods = -np.ones((Ntoks, self.num_classes, len(self.methods)))

        preds_nocrowd = -np.ones((self.N_nocrowd, len(self.methods)))
        probs_allmethods_nocrowd = -np.ones((self.N_nocrowd, self.num_classes, len(self.methods)))

        # timestamp for when we started a run. Can be compared to file versions to check what was run.
        timestamp = datetime.datetime.now().strftime('started-%Y-%m-%d-%H-%M-%S')

        for method_idx, method in enumerate(self.methods):

            print('Running method: %s' % method)

            Nseen = 0
            niter = 0
            if active_learning:
                # get the number of labels to select each iteration
                self.batch_size = int(np.ceil(AL_batch_fraction * Nannos))
                np.random.seed(self.seed)  # for repeating with different methods with same initial set
                self.annos = None
                selected_docs, selected_toks, nselected_by_doc = self._uncertainty_sampling(
                    np.ones(Ndocs, dtype=float) / self.num_classes, None, None)
            else:
                selected_docs = None
                nselected_by_doc = None

            while Nseen < Nannos and niter < max_AL_iters:
                print('Learning round %i' % niter)

                # the active learning loop
                if active_learning: # treat the unseen instances similarly to the no crowd instances -- get their posterior probabilities
                    unseen_docs = np.arange(Ndocs)
                    unseen_docs = unseen_docs[np.invert(np.in1d(unseen_docs, selected_docs))]

                    unselected_toks = np.argwhere(np.in1d(np.cumsum(self.doc_start_test) - 1, unseen_docs)).flatten()
                    self.N_unseentoks = len(unselected_toks)

                    doc_start_unseen = self.doc_start_test[unselected_toks]
                    text_unseen = self.text_test[unselected_toks]
                else:
                    self.N_unseentoks = 0
                    doc_start_unseen = None
                    text_unseen = None

                agg, probs, most_likely_seq_probs, agg_nocrowd, probs_nocrowd, agg_unseen, probs_unseen = \
                    self._run_method(method, timestamp, doc_start_unseen, text_unseen, selected_docs, nselected_by_doc,
                                     new_data if niter == 0 else False)

                if not active_learning:  # if we are training on all data at once, we can cache the BSC and HMMCrowd
                    # results so that any subsequent +LSTM steps can be done more quickly. This makes no sense for AL
                    self.aggs[method] = agg
                    self.probs[method] = probs

                if np.any(self.gold != -1):  # don't run this in the case that crowd-labelled data has no gold labels

                    if active_learning and len(agg) < len(self.gold):
                        agg_all = np.ones(len(self.gold))
                        agg_all[selected_toks] = agg.flatten()
                        agg_all[unselected_toks] = agg_unseen.flatten()
                        agg = agg_all

                        probs_all = np.zeros((len(self.gold), self.num_classes))
                        probs_all[selected_toks, :] = probs
                        probs_all[unselected_toks, :] = probs_unseen
                        probs = probs_all

                    self.calculate_experiment_scores(agg, probs, method_idx)
                    preds[:, method_idx] = agg.flatten()
                    probs_allmethods[:, :, method_idx] = probs

                if self.N_nocrowd > 0:
                    self.calculate_nocrowd_scores(agg_nocrowd, probs_nocrowd, method_idx)

                    preds_nocrowd[:, method_idx] = agg_nocrowd.flatten()
                    probs_allmethods_nocrowd[:, :, method_idx] = probs_nocrowd

                print('...done')

                # Save the results so far after each method has completed.
                Nseen = np.sum(self.annos[self.doc_start.flatten() == 1] != -1) # update the number of documents processed so far
                print('Nseen = %i' % Nseen)

                # change the timestamps to include AL loop numbers
                if save_with_timestamp:
                    file_id = timestamp + ('-Nseen%i' % Nseen)
                else:
                    file_id = ''

                _append_to_csv(self.outputdir, method, method_idx, self.scores, 'result', file_id)
                _append_to_csv(self.outputdir, method, method_idx, self.score_std, 'result_std', file_id)
                _append_to_csv(self.outputdir, method, method_idx, preds, 'pred', file_id)
                if self.N_nocrowd > 0:
                    _append_to_csv(self.outputdir, method, method_idx, self.scores_nocrowd, 'result_nocrowd', file_id)
                    _append_to_csv(self.outputdir, method, method_idx, self.score_std_nocrowd, 'result_std_nocrowd', file_id)
                    _append_to_csv(self.outputdir, method, method_idx, preds_nocrowd, 'pred_nocrowd', file_id)

                with open(os.path.join(self.outputdir, 'probs_%s.pkl' % file_id), 'wb') as fh:
                    pickle.dump(probs_allmethods, fh)

                if self.N_nocrowd > 0:
                    with open(os.path.join(self.outputdir, 'probs_nocrowd_%s.pkl' % file_id), 'wb') as fh:
                        pickle.dump(probs_allmethods_nocrowd, fh)

                # if model is not None and not active_learning and return_model:
                #     with open(self.outputdir + 'model_%s.pkl' % method, 'wb') as fh:
                #         pickle.dump(model, fh)

                if active_learning and Nseen < Nannos:
                    # non-sequential methods just provide independent label probabilities.
                    if most_likely_seq_probs is None:
                        most_likely_seq_probs = [np.prod(seq_prob) for seq_prob in np.split(probs, self.doc_start.flatten())]

                    selected_docs, selected_toks, nselected_by_doc = self._uncertainty_sampling(
                        most_likely_seq_probs, selected_toks, selected_docs)

                    print('**** Active learning: No. annos = %i' % self.annos.shape[0])
                else:
                    selected_docs = None
                niter += 1

        return self.scores, preds, probs_allmethods, \
               self.scores_nocrowd, preds_nocrowd, probs_allmethods_nocrowd


    def _run_method(self, method, timestamp, doc_start_unseen, text_unseen, selected_docs, nselected_by_doc, new_data):
        # Initialise the results because not all methods will fill these in
        most_likely_seq_probs = None # some methods can compute this
        agg_unseen = np.zeros(self.N_unseentoks)
        # default maximum entropy situation
        probs_unseen = np.ones((self.N_unseentoks, self.num_classes), dtype=float) / self.num_classes

        agg_nocrowd = np.zeros(self.N_nocrowd)
        probs_nocrowd = np.ones((self.N_nocrowd, self.num_classes))

        if method.split('_')[0] == 'best':
            agg, probs = self._run_best_worker()

        elif method.split('_')[0] == 'worst':
            agg, probs = self._run_worst_worker()

        elif method.split('_')[0] == 'clustering':
            agg, probs = self._run_clustering()

        elif method.split('_')[0] == 'majority':
            agg, probs = majority_voting.MajorityVoting(self.annos, self.num_classes).vote()

        elif method.split('_')[0] == 'mace':
            agg, probs = self._run_mace(timestamp)

        elif method.split('_')[0] == 'ds':
            agg, probs = self._run_ds()

        elif method.split('_')[0] == 'ibcc':
            agg, probs = self._run_ibcc()

        elif method.split('_')[0] == 'ibcc2':  # this is the newer and simpler implementation
            agg, probs = self._run_ibcc2()

        elif method.split('_')[0] == 'ibcc3': # use BSC with all sequential stuff switched off
            agg, probs = self._run_ibcc3()

        elif method.split('_')[0] == 'heatmapbcc': # use BSC with all sequential stuff switched off, GP on
            agg, probs = self._run_heatmapBCC()

        elif method.split('_')[0] == 'bsc' or method.split('_')[0] == 'bac':

            core_method = method.replace('_thenLSTM', '')  # the BSC method without the subsequent LSTM

            if core_method not in self.aggs:
                agg, probs, most_likely_seq_probs, agg_nocrowd, probs_nocrowd, agg_unseen, probs_unseen \
                    = self._run_bsc(
                        method,
                        doc_start_unseen=doc_start_unseen,
                        text_unseen=text_unseen
                    )
            else:
                agg = self.aggs[core_method]
                probs = self.probs[core_method]

        elif 'HMM_crowd' in method:
            if 'HMM_crowd' not in self.aggs:
                # we pass all annos here so they can be saved and reloaded from a single file in HMMCrowd
                # format, then the relevant subset selected from that.
                agg, probs, model = self._run_hmmcrowd(selected_docs, nselected_by_doc, overwrite_data_file=new_data)
                most_likely_seq_probs = model.res_prob
            else:
                agg = self.aggs['HMM_crowd']
                probs = self.probs['HMM_crowd']

        elif 'gt' in method:
            agg = self.gold.flatten()

            probs = np.zeros((len(self.gold), self.num_classes))
            probs[range(len(agg)), agg.astype(int)] = 1.0

        if '_thenLSTM' in method:
            agg, probs, agg_nocrowd, probs_nocrowd, agg_unseen, probs_unseen = self._run_LSTM(
                agg, timestamp, doc_start_unseen, text_unseen)

        return agg, probs, most_likely_seq_probs, agg_nocrowd, probs_nocrowd, agg_unseen, probs_unseen


    def calculate_experiment_scores(self, agg, probs, method_idx):
        if self.scores is None:
            self.scores = np.zeros((len(SCORE_NAMES), len(self.methods)))
            self.score_std = np.zeros((len(SCORE_NAMES) - 3, len(self.methods)))

        self.scores[:, method_idx][:, None], self.score_std[:, method_idx] = \
            calculate_scores(self.postprocess, agg, self.gold.flatten(), probs, self.doc_start_all, self.bootstrapping,
                             print_per_class_results=True
                             )


    def calculate_nocrowd_scores(self, agg, probs, method_idx):
        if self.scores_nocrowd is None:
            # for the additional test set with no crowd labels
            self.scores_nocrowd = np.zeros((len(SCORE_NAMES), len(self.methods)))
            self.score_std_nocrowd = np.zeros((len(SCORE_NAMES) - 3, len(self.methods)))

        self.scores_nocrowd[:,method_idx][:,None], self.score_std_nocrowd[:, method_idx] = calculate_scores(
            self.postprocess, agg, self.gold_nocrowd.flatten(), probs, self.doc_start_nocrowd,
            self.bootstrapping, print_per_class_results=True)


    # Methods -----------------------------------------------------------------

    def _run_best_worker(self):
        # choose the best classifier by f1-score
        f1scores = np.zeros_like(self.annos) - 1.0
        print('F1 scores for individual workers:')

        individual_scores = []

        for w in range(self.annos.shape[1]):

            valididxs = self.annos[:, w] != -1

            if not np.any(valididxs):
                continue

            f1_by_class = skm.f1_score(self.gold.flatten()[valididxs], self.annos[valididxs, w], labels=range(self.num_classes),
                                       average=None)
            f1_w = np.mean(f1_by_class[np.unique(self.gold[valididxs]).astype(int)])

            #print(f1_w)
            individual_scores.append(f1_w)

            f1scores[valididxs, w] = f1_w

        #print(sorted(individual_scores))

        best_idxs = np.argmax(f1scores, axis=1)
        agg = self.annos[np.arange(self.annos.shape[0]), best_idxs]
        probs = np.zeros((self.gold.shape[0], self.num_classes))
        for k in range(self.gold.shape[0]):
            probs[k, int(agg[k])] = 1

        return agg, probs

    def _run_worst_worker(self):
        # choose the weakest classifier by f1-score
        f1scores = np.zeros_like(self.annos) + np.inf
        for w in range(self.annos.shape[1]):

            valididxs = self.annos[:, w] != -1

            if not np.any(valididxs):
                continue

            f1_by_class = skm.f1_score(self.gold.flatten()[valididxs], self.annos[valididxs, w], labels=range(self.num_classes),
                                       average=None)
            f1scores[valididxs, w] = np.mean(f1_by_class[np.unique(self.gold[valididxs]).astype(int)])

        worst_idxs = np.argmin(f1scores, axis=1)
        agg = self.annos[np.arange(self.annos.shape[0]), worst_idxs]

        probs = np.zeros((self.gold.shape[0], self.num_classes))
        for k in range(self.gold.shape[0]):
            probs[k, int(agg[k])] = 1

        return agg, probs

    def _run_clustering(self):
        cl = clustering.Clustering(self.gold, self.annos, self.doc_start)
        agg = cl.run()

        probs = np.zeros((self.gold.shape[0], self.num_classes))
        for k in range(self.gold.shape[0]):
            probs[k, int(agg[k])] = 1

        return agg, probs

    def _run_ds(self):
        probs = ds(self.annos, self.num_classes, self.beta0_factor, self.max_iter)
        agg = np.argmax(probs, axis=1)
        return agg, probs

    def _run_heatmapBCC(self):
        bsc_model = HeatmapBCC(L=self.num_classes, K=self.annos.shape[1], max_iter=self.max_iter,  # eps=-1,
                     alpha0_diags=self.alpha0_diags, alpha0_factor=self.alpha0_factor, beta0_factor=self.beta0_factor,
                     use_lowerbound=False)
        bsc_model.verbose = True

        if self.opt_hyper:
            np.random.seed(592)  # for reproducibility
            probs, agg = bsc_model.optimize(self.annos, self.doc_start, self.text, maxfun=1000)
        else:
            probs, agg, pseq = bsc_model.fit_predict(self.annos, self.doc_start, self.text)

        return agg, probs

    def _run_ibcc3(self):
        bsc_model = IBCC(L=self.num_classes, K=self.annos.shape[1], max_iter=self.max_iter,  # eps=-1,
                     alpha0_diags=self.alpha0_diags, alpha0_factor=self.alpha0_factor, beta0_factor=self.beta0_factor,
                     use_lowerbound=False)
        bsc_model.verbose = True

        if self.opt_hyper:
            np.random.seed(592)  # for reproducibility
            probs, agg = bsc_model.optimize(self.annos, self.doc_start, self.text, maxfun=1000)
        else:
            probs, agg, pseq = bsc_model.fit_predict(self.annos, self.doc_start, self.text)

        return agg, probs

    def _run_ibcc2(self):
        probs, _ = ibccvb(self.annos, self.num_classes, self.beta0_factor,
                          self.alpha0_factor, self.alpha0_diags, self.begin_factor, self.max_iter)
        agg = np.argmax(probs, axis=1)
        return agg, probs

    def _run_ibcc(self, use_ml=False):
        if use_ml:
            alpha0 = np.ones((self.num_classes, self.num_classes)) + 0.1 # no prior information at all, just add a small regularization term
            ibc = ibcc.IBCC(nclasses=self.num_classes, nscores=self.num_classes, nu0=np.ones(self.num_classes),
                        alpha0=alpha0, uselowerbound=True, use_ml=True)
        else:
            self.ibcc_beta0 = np.ones(self.num_classes) * self.beta0_factor
            self.ibcc_alpha0 = (self.alpha0_factor/float(self.num_classes-1)) * np.ones((self.num_classes, self.num_classes)) \
                               + (self.alpha0_diags + self.alpha0_factor *(1-1/float(self.num_classes-1))) * np.eye(self.num_classes)

            ibc = ibcc.IBCC(nclasses=self.num_classes, nscores=self.num_classes, nu0=self.ibcc_beta0,
                        alpha0=self.ibcc_alpha0, uselowerbound=True)

        ibc.verbose = True
        ibc.max_iterations = self.max_iter
        # ibc.optimise_alpha0_diagonals = True

        if self.opt_hyper:
            probs = ibc.combine_classifications(self.annos, table_format=True, optimise_hyperparams=True,
                                                maxiter=10000)
        else:
            probs = ibc.combine_classifications(self.annos, table_format=True)  # posterior class probabilities
        agg = probs.argmax(axis=1)  # aggregated class labels

        return agg, probs

    def _run_mace(self, timestamp):
        anno_path = os.path.join(self.outputdir, 'annos_tmp_%s' % timestamp)

        annotations = pd.DataFrame(self.annos)
        annotations.replace(-1, np.nan)
        annotations.to_csv(anno_path, sep=',', header=False, index=False)

        subprocess.call(['java', '-jar', './src/baselines/MACE/MACE.jar', '--distribution', '--prefix',
                        os.path.join(self.outputdir, 'mace'),
                        anno_path])  # , stdout = devnull, stderr = devnull)

        result = np.genfromtxt(os.path.join(self.outputdir, 'mace.prediction'))

        probs = np.zeros((self.gold.shape[0], self.num_classes))
        for i in range(result.shape[0]):
            for j in range(0, self.num_classes * 2, 2):
                probs[i, int(result[i, j])] = result[i, j + 1]

        os.remove(anno_path)  # clean up tmp file

        agg = np.argmax(probs, 1)
        return agg, probs

    def _run_bsc(self, method, doc_start_unseen=None, text_unseen=None):

        method_bits = method.split('_')

        if method_bits[-1] == 'noHMM':
            transition_model = 'None'
        elif method_bits[-1] == 'GP':
            transition_model = 'GP'
        else:
            transition_model = 'HMM'

        # needs to run integrate method for task 2 as well
        if len(method_bits) > 2 and 'integrateLSTM' in method_bits:
            if len(method_bits) > 3 and 'atEnd' in method_bits:
                use_LSTM = 2
            else:
                use_LSTM = 1
        else:
            use_LSTM = 0

        if len(method_bits) > 2 and 'integrateIF' in method_bits:
            use_IF = True
        else:
            use_IF = False

        if len(method_bits) > 2 and 'integrateGP' in method_bits:
            if len(method_bits) > 3 and 'atEnd' in method_bits:
                use_GP = 2
            else:
                use_GP = 1
        else:
            use_GP = 0

        worker_model = method_bits[1]

        L = self.num_classes

        num_types = (self.num_classes - 1) / 2
        outside_label = 1
        inside_labels = (np.arange(num_types) * 2 + 1).astype(int)
        inside_labels[0] = 0
        begin_labels = (np.arange(num_types) * 2 + 2).astype(int)

        data_model = []
        dev_sentences = []

        if use_LSTM > 0:
            if self.crf_probs:
                data_model.append('LSTM_crf')
            else:
                data_model.append('LSTM')

            if self.gold_val is not None and self.doc_start_val is not None and self.text_val is not None:
                dev_sentences, _, _ = lstm_wrapper.data_to_lstm_format(self.text_val,
                                                                       self.doc_start_val, self.gold_val)

        if use_GP:
            data_model.append('GP')

        bsc_model = bayesian_combination.BC(L=L, K=self.annos.shape[1], max_iter=self.max_iter,  # eps=-1,
                        inside_labels=inside_labels, outside_label=outside_label, beginning_labels=begin_labels,
                        alpha0_diags=self.alpha0_diags, alpha0_factor=self.alpha0_factor, alpha0_B_factor=self.begin_factor,
                        beta0_factor=self.beta0_factor, annotator_model=worker_model, tagging_scheme='IOB2',
                        taggers=data_model, true_label_model=transition_model, discrete_feature_likelihoods=use_IF,
                        use_lowerbound=False, model_dir=self.outputdir, converge_workers_first=(use_LSTM==2) or (use_GP==2))
        bsc_model.verbose = True
        bsc_model.max_internal_iters = self.max_internal_iter

        if self.opt_hyper:
            np.random.seed(592)  # for reproducibility
            probs, agg = bsc_model.optimize(self.annos, self.doc_start, self.text, maxfun=1000, dev_sentences=dev_sentences)
        else:
            probs, agg, pseq = bsc_model.fit_predict(self.annos, self.doc_start, self.text, dev_sentences=dev_sentences)

        if self.gold_nocrowd is not None and '_thenLSTM' not in method:
            probs_nocrowd, agg_nocrowd = bsc_model.predict(self.doc_start_nocrowd, self.text_nocrowd)
        else:
            probs_nocrowd = None
            agg_nocrowd = None

        if '_thenLSTM' not in method and doc_start_unseen is not None and len(doc_start_unseen) > 0:
            probs_unseen, agg_unseen = bsc_model.predict(doc_start_unseen, text_unseen)
        else:
            probs_unseen = None
            agg_unseen = None

        return agg, probs, pseq, agg_nocrowd, probs_nocrowd, agg_unseen, probs_unseen

    def _run_hmmcrowd(self, doc_subset, nselected_by_doc,
                      overwrite_data_file=False):

        sentences, crowd_labels, nfeats = data_to_hmm_crowd_format(self.annos_all, self.text_all, self.doc_start_all,
                            self.outputdir, overwrite=overwrite_data_file)
        sentences, crowd_labels = subset_hmm_crowd_data(sentences, crowd_labels, doc_subset, nselected_by_doc)

        data = crowd_data(sentences, crowd_labels)
        hc = HMM_crowd(self.num_classes, nfeats, data, None, None, n_workers=self.annos_all.shape[1],
                       vb=[self.beta0_factor, self.alpha0_factor], smooth=self.alpha0_factor)
        hc.init(init_type='dw', wm_rep='cv', dw_em=5, wm_smooth=self.alpha0_factor)

        print('Running HMM-crowd inference...')
        hc.em(self.max_iter) # performance goes down with more iterations...?!

        print('Computing most likely sequence...')
        hc.mls()

        print('HMM-crowd complete.')
        # agg = np.array(hc.res).flatten()
        agg = np.concatenate(hc.res)[:, None]
        probs = []
        for sentence_post_arr in hc.sen_posterior:
            for tok_post_arr in sentence_post_arr:
                probs.append(tok_post_arr)
        probs = np.array(probs)

        return agg.flatten(), probs, hc

    def _run_LSTM(self, train_labs, timestamp, doc_start_unseen=None, text_unseen=None):

        valididxs = np.any(self.annos != -1, axis=1)
        labelled_sentences, IOB_map, IOB_label = lstm_wrapper.data_to_lstm_format(
            self.text[valididxs], self.doc_start[valididxs], train_labs.flatten()[valididxs], self.num_classes)

        np.random.seed(592) # for reproducibility

        if self.gold_val is None or self.doc_start_val is None or self.text_val is None:
            # If validation set is unavailable, select a random subset of combined data to use for validation
            # Simulates a scenario where all we have available are crowd labels.
            train_sentences, dev_sentences, self.gold_val = lstm_wrapper.split_train_to_dev(labelled_sentences)
            all_sentences = labelled_sentences
        else:
            train_sentences = labelled_sentences
            dev_sentences, _, _ = lstm_wrapper.data_to_lstm_format(self.text_val,
                                                                   self.doc_start_val, self.gold_val)

            all_sentences = np.concatenate((train_sentences, dev_sentences), axis=0)

        lstm = lstm_wrapper.LSTMWrapper(os.path.join(self.outputdir, 'models_LSTM_%s' % timestamp))

        print('Running LSTM with crf probs = %s' % self.crf_probs)

        lstm.train_LSTM(all_sentences, train_sentences, dev_sentences, self.gold_val, IOB_map,
                        IOB_label, self.num_classes, freq_eval=5, n_epochs=self.max_internal_iter,
                        crf_probs=self.crf_probs, max_niter_no_imprv=2)

        # now make predictions for all sentences
        test_sentences, _, _ = lstm_wrapper.data_to_lstm_format(
            self.text, self.doc_start, np.ones(len(train_labs)), self.num_classes)
        agg, probs = lstm.predict_LSTM(test_sentences)

        if self.N_nocrowd > 0:
            test_sentences, _, _ = lstm_wrapper.data_to_lstm_format(
                self.text_nocrowd, self.doc_start_nocrowd, np.ones(self.N_nocrowd), self.num_classes)

            agg_nocrowd, probs_nocrowd = lstm.predict_LSTM(test_sentences)
        else:
            agg_nocrowd = None
            probs_nocrowd = None

        if doc_start_unseen is not None:
            N_unseen = len(doc_start_unseen)
            test_sentences, _, _ = lstm_wrapper.data_to_lstm_format(
                text_unseen, doc_start_unseen, np.ones(N_unseen), self.num_classes)

            agg_unseen, probs_unseen = lstm.predict_LSTM(test_sentences)
        else:
            agg_unseen = None
            probs_unseen = None

        return agg, probs, agg_nocrowd, probs_nocrowd, agg_unseen, probs_unseen

    def _uncertainty_sampling(self, most_likely_probs, selected_toks, selected_docs):

        unfinished_toks = np.ones(self.annos_test.shape[0], dtype=bool)

        if self.annos is not None:
            unfinished_toks[selected_toks] = (np.sum(self.annos_test[selected_toks] != -1, axis=1) - np.sum(self.annos != -1, axis=1)) > 0

        unfinished_toks = np.sort(np.argwhere(unfinished_toks).flatten())

        # random selection
        #new_selection = np.random.choice(unseen_docs, batch_size, replace=False)

        # probs = probs[unfinished_toks, :]
        # negentropy = np.log(probs) * probs
        # negentropy[probs == 0] = 0
        # negentropy = np.sum(negentropy, axis=1)

        docids_by_tok = (np.cumsum(self.doc_start_test) - 1)[unfinished_toks]

        if self.random_sampling:
            new_selected_docs = np.random.choice(np.unique(docids_by_tok), self.batch_size, replace=False)

        else:
            # Ndocs = np.max(docids_by_tok) + 1
            # negentropy_docs = np.zeros(Ndocs, dtype=float)
            # count_unseen_toks = np.zeros(Ndocs, dtype=float)
            #
            # # now sum up the entropy for each doc and normalise by length (otherwise we'll never label the short ones)
            # for i, _ in enumerate(unfinished_toks):
            #     # find doc ID for this tok
            #     docid = docids_by_tok[i]
            #
            #     negentropy_docs[docid] += negentropy[i]
            #     count_unseen_toks[docid] += 1
            #
            # negentropy_docs /= count_unseen_toks
            #
            # # assume that batch size is less than number of docs...
            # new_selected_docs = np.argsort(negentropy_docs)[:batch_size]
            new_selected_docs = np.argsort(most_likely_probs)[:self.batch_size]

        new_selected_toks = np.in1d(np.cumsum(self.doc_start_test) - 1, new_selected_docs)

        new_label_count = np.zeros(self.annos_test.shape[0])
        new_label_count[new_selected_toks] += 1

        # add to previously selected toks and docs
        if self.annos is not None:
            # how many labels do we have for these tokens so far?
            current_label_count = np.sum(self.annos != -1, axis=1)

            # add one for the new round
            new_label_count[selected_toks] += current_label_count

            selected_docs = np.sort(np.unique(np.concatenate((selected_docs, new_selected_docs))))

            new_selected_toks[selected_toks] = True
            selected_toks = new_selected_toks
        else:
            selected_toks = new_selected_toks
            selected_docs = new_selected_docs

        nselected_by_doc = new_label_count[selected_toks]
        nselected_by_doc = nselected_by_doc[self.doc_start_test[selected_toks].flatten() == 1]

        # find the columns for each token that we should get from the full set of crowdsource annos
        mask = np.cumsum(self.annos_test[selected_toks] != -1, axis=1) <= new_label_count[selected_toks][:, None]

        # make a new label set
        annotations = np.zeros((np.sum(selected_toks), self.annos_test.shape[1])) - 1

        # copy in the data from self.annos_test
        annotations[mask] = self.annos_test[selected_toks, :][mask]
        self.annos = annotations

        self.doc_start = self.doc_start_test[selected_toks]
        self.text = self.text_test[selected_toks]

        return selected_docs, np.argwhere(selected_toks).flatten(), nselected_by_doc
