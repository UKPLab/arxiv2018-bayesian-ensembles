'''
Created on Nov 1, 2016

@author: Melvin Laux
'''

import matplotlib.pyplot as plt
import logging

from evaluation.span_level_f1 import precision, recall, f1

logging.basicConfig(level=logging.DEBUG)

import pickle
import datetime
from baselines import ibcc, clustering, majority_voting
from algorithm import bac
from data import data_utils
import configparser
import os, subprocess
import sklearn.metrics as skm
import numpy as np
import evaluation.metrics as metrics
import glob
from baselines.hmm import HMM_crowd
from baselines.util import crowd_data, crowdlab, instance

# things needed for the LSTM
import lample_lstm_tagger.lstm_wrapper as lstm_wrapper

class Experiment(object):
    '''
    classdocs
    '''
    
    generator = None
    
    config_file = None
    
    param_values = None
    param_idx = None
    param_fixed = None
    
    acc_bias = None
    miss_bias = None
    short_bias = None
    
    PARAM_NAMES = ['acc_bias',
                   'miss_bias',
                   'short_bias',
                   'num_docs',
                   'doc_length',
                   'group_sizes'
                   ]

    SCORE_NAMES = ['accuracy',
                   'precision-tokens',
                   'recall-tokens',
                   'f1-score-tokens',
                   'auc-score',
                   'cross-entropy-error',
                   'precision-spans-strict',
                   'recall-spans-strict',
                   'f1-score-spans-strict',
                   'precision-spans-relaxed',
                   'recall-spans-relaxed',
                   'f1-score-spans-relaxed',
                   'count error',
                   'number of invalid labels',
                   'mean length error'
                   ]
    
    generate_data= False
    
    methods = None

    alpha0 = None
    nu0 = None
    
    exclusions = {}

    num_runs = None
    doc_length = None
    group_sizes = None
    num_docs = None
    
    num_classes = None
    
    output_dir = '/output/'
    
    save_results = False
    save_plots = False
    show_plots = False
    
    postprocess = False # previous papers did not use this so we leave out to make results comparable.
    # Only simple IOB2 is implemented so far.

    opt_hyper = False

    def __init__(self, generator, nclasses, nannotators, config=None,
                 alpha0_factor=16.0, alpha0_diags = 1.0, nu0_factor = 100.0):

        self.generator = generator
        
        if not (config == None):
            self.config_file = config
            self.read_config_file()

        print('setting initial priors for Bayesian methods...')

        self.num_classes = nclasses
        self.nannotators = nannotators

        self.alpha0_factor = alpha0_factor
        self.alpha0_diags = alpha0_diags
        self.nu0_factor = nu0_factor

        self.bac_nu0 = np.ones((nclasses + 1, nclasses)) * self.nu0_factor
        self.ibcc_nu0 = np.ones(nclasses) * self.nu0_factor

        # save results from methods here. If we use compound methods, we can reuse these results in different
        # combinations of methods.
        self.aggs = {}
        self.probs = {}
            
    def read_config_file(self):
        
        print('Reading experiment config file...')
        
        parser = configparser.ConfigParser()
        parser.read(self.config_file)
    
        # set up parameters
        parameters = dict(parser.items('parameters'))
        self.param_idx = int(parameters['idx'].split('#')[0].strip())
        self.param_values = np.array(eval(parameters['values'].split('#')[0].strip()))
        
        self.acc_bias = np.array(eval(parameters['acc_bias'].split('#')[0].strip()))
        self.miss_bias = np.array(eval(parameters['miss_bias'].split('#')[0].strip()))
        self.short_bias = np.array(eval(parameters['short_bias'].split('#')[0].strip()))
        
        
        self.methods = eval(parameters['methods'].split('#')[0].strip())
        
        self.postprocess = eval(parameters['postprocess'].split('#')[0].strip())
        
        print(self.methods)
                
        self.num_runs = int(parameters['num_runs'].split('#')[0].strip())
        self.doc_length = int(parameters['doc_length'].split('#')[0].strip())
        self.group_sizes = np.array(eval(parameters['group_sizes'].split('#')[0].strip()))
        self.num_classes = int(parameters['num_classes'].split('#')[0].strip())
        self.num_docs = int(parameters['num_docs'].split('#')[0].strip())
        self.generate_data = eval(parameters['generate_data'].split('#')[0].strip())
        
        # set up output
        parameters = dict(parser.items('output'))
        self.output_dir = parameters['output_dir']
        self.save_results = eval(parameters['save_results'].split('#')[0].strip())
        self.save_plots = eval(parameters['save_plots'].split('#')[0].strip())
        self.show_plots = eval(parameters['show_plots'].split('#')[0].strip())

    def tune_alpha0(self, alpha0diag_propsals, alpha0factor_proposals, method, annotations, ground_truth, doc_start,
                    outputdir, text, anno_path=None, metric_idx_to_optimise=8):

        self.methods = [method]

        scores = np.zeros((len(alpha0diag_propsals), len(alpha0factor_proposals) ))

        best_scores = np.zeros(3)

        for i, alpha0diag in enumerate(alpha0diag_propsals):

            self.alpha0_diags = alpha0diag

            for j, alpha0factor in enumerate(alpha0factor_proposals):

                # reset saved data so that models are run again.
                self.aggs = {}
                self.probs = {}

                outputdir_ij = outputdir + ('_%i_%i_' % (i, j)) + method + '/'

                self.alpha0_factor = alpha0factor

                all_scores, _, _, _, _, _ = self.run_methods(annotations, ground_truth, doc_start, outputdir_ij, text,
                                                             anno_path, bootstrapping=False)
                scores[i, j] = all_scores[metric_idx_to_optimise, :] # 3 is F1score
                print('Scores for %f, %f: %f' % (alpha0diag, alpha0factor, scores[i, j]))

                if scores[i, j] > best_scores[0]:
                    best_scores[0] = scores[i, j]
                    best_scores[1] = i
                    best_scores[2] = j

                print('Saving scores for this setting to %s' % (outputdir + '/%s_scores.csv' % method))
                np.savetxt(outputdir + '/%s_scores.csv' % method, scores, fmt='%s', delimiter=',',
                           header=str(self.methods).strip('[]'))

                np.savetxt(outputdir + '/%s_bestscores.csv' % method, best_scores, fmt='%s', delimiter=',',
                           header=str(self.methods).strip('[]'))

        return best_scores

    def _run_best_worker(self, annos, gt, doc_start):
        # choose the best classifier by f1-score
        f1scores = np.zeros(annos.shape[1])
        for w in range(annos.shape[1]):
            f1scores[w] = skm.f1_score(gt.flatten(), annos[:, w], average='macro')
        f1scores = f1scores[None, :]
        f1scores = np.tile(f1scores, (annos.shape[0], 1))

        f1scores[annos == -1] = -1  # remove the workers that did not label those particular annotations

        best_idxs = np.argmax(f1scores, axis=1)
        agg = annos[np.arange(annos.shape[0]), best_idxs]
        probs = np.zeros((gt.shape[0], self.num_classes))
        for k in range(gt.shape[0]):
            probs[k, int(agg[k])] = 1

        return agg, probs

    def _run_worst_worker(self, annos, gt, doc_start):
        # choose the weakest classifier by f1-score
        f1scores = np.zeros(annos.shape[1])
        for w in range(annos.shape[1]):
            f1scores[w] = skm.f1_score(gt.flatten(), annos[:, w], average='macro')
        f1scores = f1scores[None, :]
        f1scores = np.tile(f1scores, (annos.shape[0], 1))

        f1scores[annos == -1] = np.inf  # remove the workers that did not label those particular annotations

        worst_idxs = np.argmin(f1scores, axis=1)
        agg = annos[np.arange(annos.shape[0]), worst_idxs]

        probs = np.zeros((gt.shape[0], self.num_classes))
        for k in range(gt.shape[0]):
            probs[k, int(agg[k])] = 1

        return agg, probs

    def _run_clustering(self, annotations, doc_start, ground_truth):
        cl = clustering.Clustering(ground_truth, annotations, doc_start)
        agg = cl.run()

        probs = np.zeros((ground_truth.shape[0], self.num_classes))
        for k in range(ground_truth.shape[0]):
            probs[k, int(agg[k])] = 1

        return agg, probs

    def _run_mace(self, anno_path, tmp_path, ground_truth):
        # devnull = open(os.devnull, 'w')
        #subprocess.call(['java', '-jar', './MACE/MACE.jar', '--distribution', '--prefix',
        #                 tmp_path + '/mace',
        #                 anno_path])  # , stdout = devnull, stderr = devnull)

        result = np.genfromtxt(tmp_path + '/mace.prediction')

        probs = np.zeros((ground_truth.shape[0], self.num_classes))
        for i in range(result.shape[0]):
            for j in range(0, self.num_classes * 2, 2):
                probs[i, int(result[i, j])] = result[i, j + 1]

        agg = np.argmax(probs, 1)

        return agg, probs

    def _run_ibcc(self, annotations, use_ml=False):
        if use_ml:
            alpha0 = 0.0000001 * np.ones((self.num_classes, self.num_classes)) # no prior information at all
            ibc = ibcc.IBCC(nclasses=self.num_classes, nscores=self.num_classes, nu0=self.ibcc_nu0,
                        alpha0=alpha0, uselowerbound=True, use_ml=True)
        else:
            self.ibcc_alpha0 = self.alpha0_factor * np.ones((self.num_classes, self.num_classes)) \
                               + self.alpha0_diags * np.eye(self.num_classes)
            ibc = ibcc.IBCC(nclasses=self.num_classes, nscores=self.num_classes, nu0=self.ibcc_nu0,
                        alpha0=self.ibcc_alpha0, uselowerbound=False)

        ibc.verbose = True
        ibc.max_iterations = 20
        # ibc.optimise_alpha0_diagonals = True

        if self.opt_hyper:
            probs = ibc.combine_classifications(annotations, table_format=True, optimise_hyperparams=True,
                                                maxiter=10000)
        else:
            probs = ibc.combine_classifications(annotations, table_format=True)  # posterior class probabilities
        agg = probs.argmax(axis=1)  # aggregated class labels

        return agg, probs, ibc

    def _run_bac(self, annotations, doc_start, text, method, use_LSTM=0, use_BOF=False,
                 ground_truth_val=None, doc_start_val=None, text_val=None,
                 ground_truth_nocrowd=None, doc_start_nocrowd=None, text_nocrowd=None, transition_model='HMM'):

        self.bac_worker_model = method.split('_')[1]
        L = self.num_classes

        # matrices are repeated for the different annotators/previous label conditions inside the BAC code itself.
        if self.bac_worker_model == 'seq' or self.bac_worker_model == 'ibcc' or self.bac_worker_model == 'vec':

            alpha0_factor = self.alpha0_factor / ((L-1)/2)
            alpha0_diags = self.alpha0_diags + self.alpha0_factor - alpha0_factor

            self.bac_alpha0 = alpha0_factor * np.ones((L, L)) + \
                              alpha0_diags * np.eye(L)

        elif self.bac_worker_model == 'mace':
            alpha0_factor = self.alpha0_factor #/ ((L-1)/2)
            self.bac_alpha0 = alpha0_factor * np.ones((2 + L))
            self.bac_alpha0[1] += self.alpha0_diags  # diags are bias toward correct answer

        elif self.bac_worker_model == 'acc':
            self.bac_alpha0 = self.alpha0_factor * np.ones((2))
            self.bac_alpha0[1] += self.alpha0_diags  # diags are bias toward correct answer

        num_types = (self.num_classes - 1) / 2
        outside_labels = [-1, 1]
        inside_labels = (np.arange(num_types) * 2 + 1).astype(int)
        inside_labels[0] = 0
        begin_labels = (np.arange(num_types) * 2 + 2).astype(int)

        if use_LSTM and ground_truth_val is not None and doc_start_val is not None and text_val is not None:
            dev_sentences, _, _ = lstm_wrapper.data_to_lstm_format(len(ground_truth_val), text_val,
                                                                   doc_start_val, ground_truth_val)
        else:
            dev_sentences = None

        data_model = []

        if use_LSTM > 0:
            data_model.append('LSTM')
        if use_BOF:
            data_model.append('IF')

        alg = bac.BAC(L=L, K=annotations.shape[1], inside_labels=inside_labels, outside_labels=outside_labels,
                      beginning_labels=begin_labels, alpha0=self.bac_alpha0,
                      nu0=self.bac_nu0 if transition_model == 'HMM' else self.ibcc_nu0,
                      exclusions=self.exclusions, before_doc_idx=1, worker_model=self.bac_worker_model,
                      tagging_scheme='IOB2',
                      data_model=data_model, transition_model=transition_model)
        alg.max_iter = 20
        alg.verbose = True

        if self.opt_hyper:
            probs, agg = alg.optimize(annotations, doc_start, text, maxfun=1000, dev_data=dev_sentences,
                                      converge_workers_first=use_LSTM==2, )
        else:
            probs, agg = alg.run(annotations, doc_start, text, dev_data=dev_sentences,
                                 converge_workers_first=use_LSTM==2, )

        model = alg

        if ground_truth_nocrowd is not None:
            probs_nocrowd, agg_nocrowd = alg.predict(doc_start_nocrowd, text_nocrowd)
        else:
            probs_nocrowd = None
            agg_nocrowd = None

        return agg, probs, model, agg_nocrowd, probs_nocrowd

    def _run_hmmcrowd(self, annotations, text, doc_start, outputdir, doc_subset, overwrite_data_file=False):
        sentences, crowd_labels, nfeats = self.data_to_hmm_crowd_format(annotations, text, doc_start, outputdir,
                                                                        doc_subset, overwrite=overwrite_data_file)

        data = crowd_data(sentences, crowd_labels)
        hc = HMM_crowd(self.num_classes, nfeats, data, None, None, n_workers=annotations.shape[1],
                       vb=[0.1, 0.1], ne=1)
        hc.init(init_type='dw', wm_rep='cv2', dw_em=5, wm_smooth=0.1)

        print('Running HMM-crowd inference...')
        hc.em(1)  # 20) # 10 is too much, performance goes down. This shouldn't really happen.

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

    def _run_LSTM(self, N_withcrowd, text, doc_start, train_labs, test_no_crowd, N_nocrowd, text_nocrowd, doc_start_nocrowd,
                  ground_truth_nocrowd, text_val, doc_start_val, ground_truth_val):
        '''
        Reasons why we need the LSTM integration:
        - Could use a simpler model like hmm_crowd for task 1
        - Performance may be better with LSTM
        - Active learning possible with simpler model, but how novel?
        - Integrating LSTM shows a generalisation -- step forward over models that fix the data
        representation
        - Can do task 2 better than a simple model like Hmm_crowd
        - Can do task 2 better than training on HMM_crowd then LSTM, or using LSTM-crowd.s
        '''
        labelled_sentences, IOB_map, _ = lstm_wrapper.data_to_lstm_format(N_withcrowd, text, doc_start,
                                                                          train_labs.flatten(), self.num_classes)

        if ground_truth_val is None or doc_start_val is None or text_val is None:
            # If validation set is unavailable, select a random subset of combined data to use for validation
            # Simulates a scenario where all we have available are crowd labels.
            train_sentences, dev_sentences, ground_truth_val = lstm_wrapper.split_train_to_dev(labelled_sentences)
            all_sentences = labelled_sentences
        else:
            print(ground_truth_val.shape)
            ground_truth_val = ground_truth_val[ground_truth_val != -1]
            print(ground_truth_val.shape)

            train_sentences = labelled_sentences
            dev_sentences, _, _ = lstm_wrapper.data_to_lstm_format(len(ground_truth_val), text_val,
                                                                   doc_start_val, ground_truth_val)


            all_sentences = np.concatenate((train_sentences, dev_sentences), axis=0)

        lstm, f_eval, _ = lstm_wrapper.train_LSTM(all_sentences, train_sentences, dev_sentences,
                                                  ground_truth_val, IOB_map, self.num_classes, n_epochs=100)

        # now make predictions for all sentences
        agg, probs = lstm_wrapper.predict_LSTM(lstm, labelled_sentences, f_eval, self.num_classes, IOB_map)

        if test_no_crowd:
            labelled_sentences, IOB_map, _ = lstm_wrapper.data_to_lstm_format(N_nocrowd, text_nocrowd,
                                                          doc_start_nocrowd, np.zeros(N_nocrowd) - 1, self.num_classes)

            agg_nocrowd, probs_nocrowd = lstm_wrapper.predict_LSTM(lstm, labelled_sentences, f_eval,
                                                                   self.num_classes, IOB_map)

        else:
            agg_nocrowd = None
            probs_nocrowd = None

        return agg, probs, agg_nocrowd, probs_nocrowd

    def _get_nocrowd_test_data(self, annotations, doc_start, ground_truth, text):

        crowd_labels_present = np.any(annotations != -1, axis=1)
        N_withcrowd = np.sum(crowd_labels_present)

        no_crowd_labels = np.invert(crowd_labels_present)
        N_nocrowd = np.sum(no_crowd_labels)

        # check whether the additional test set needs to be split from the main dataset
        if N_nocrowd == 0:
            test_no_crowd = False

            doc_start_nocrowd = None
            ground_truth_nocrowd = None
            text_nocrowd = None
        else:
            test_no_crowd = True

            annotations = annotations[crowd_labels_present, :]
            ground_truth_nocrowd = ground_truth[no_crowd_labels]
            ground_truth = ground_truth[crowd_labels_present]

            text_nocrowd = text[no_crowd_labels]
            text = text[crowd_labels_present]

            doc_start_nocrowd = doc_start[no_crowd_labels]
            doc_start = doc_start[crowd_labels_present]

        return doc_start_nocrowd, ground_truth_nocrowd, text_nocrowd, test_no_crowd, N_nocrowd, annotations, \
               doc_start, ground_truth, text, N_withcrowd

    def _uncertainty_sampling(self, annotations_all, doc_start_all, text_all, batch_size, probs, selected_docs, Ndocs):

        unseen_docs = np.ones(Ndocs, dtype=bool)
        unseen_docs[selected_docs] = False
        unseen_docs = np.argwhere(unseen_docs).flatten()

        # random selection
        #new_selection = np.random.choice(unseen_docs, batch_size, replace=False)

        probs = probs[unseen_docs, :]
        negentropy = np.log(probs) * probs
        negentropy[probs == 0] = 0
        negentropy = np.sum(negentropy, axis=1)

        most_uncertain = np.argsort(negentropy)[:batch_size]
        new_selection = unseen_docs[most_uncertain]

        selected_docs = np.concatenate((selected_docs, new_selection))

        selected_toks = np.in1d(np.cumsum(doc_start_all) - 1, selected_docs)

        annotations = annotations_all[selected_toks, :]
        doc_start = doc_start_all[selected_toks]
        text = text_all[selected_toks]

        return annotations, doc_start, text, selected_docs, selected_toks

    def run_methods(self, annotations, ground_truth, doc_start, outputdir=None, text=None,
                    ground_truth_nocrowd=None, doc_start_nocrowd=None, text_nocrowd=None,
                    ground_truth_val=None, doc_start_val=None, text_val=None,
                    return_model=False, rerun_all=False, new_data=False,
                    active_learning=False, AL_batch_fraction=0.1,
                    bootstrapping=True):
        '''
        Run the aggregation methods and evaluate them.
        :param annotations:
        :param ground_truth: class labels for computing the performance metrics. Missing values should be set to -1.
        :param doc_start:
        :param text: feature vectors.
        :param anno_path: filename containing the annotations, used by MACE only
        :param return_model:
        :param ground_truth_nocrowd: for evaluating on a second set of data points where crowd labels are not available
        :param text_nocrowd: the features for testing the model trained on crowdsourced data to classify data with no labels at all
        :param ground_truth_val: for tuning methods that predict on features only with no crowd labels, e.g. LSTM
        :return:
        '''
        # goldidxs = (ground_truth != -1).flatten()
        # annotations = annotations[goldidxs, :]
        # ground_truth = ground_truth[goldidxs]
        # doc_start = doc_start[goldidxs]

        if outputdir is not None:
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)

        print('Running experiment, Optimisation=%s' % (self.opt_hyper))

        scores_allmethods = np.zeros((len(self.SCORE_NAMES), len(self.methods)))
        score_std_allmethods = np.zeros((len(self.SCORE_NAMES)-3, len(self.methods)))

        preds_allmethods = -np.ones((annotations.shape[0], len(self.methods)))
        probs_allmethods = -np.ones((annotations.shape[0], self.num_classes, len(self.methods)))

        # a second test set with no crowd labels was supplied directly
        if ground_truth_nocrowd is not None and text_nocrowd is not None and doc_start_nocrowd is not None:
            test_no_crowd = True
            N_withcrowd = annotations.shape[0]
            N_nocrowd = ground_truth_nocrowd.shape[0]
        else:
            doc_start_nocrowd, ground_truth_nocrowd, text_nocrowd, test_no_crowd, N_nocrowd, annotations, doc_start, \
            ground_truth, text, N_withcrowd = self._get_nocrowd_test_data(annotations, doc_start, ground_truth, text)

        # for the additional test set with no crowd labels
        scores_nocrowd = np.zeros((len(self.SCORE_NAMES), len(self.methods)))
        score_std_nocrowd = np.zeros((len(self.SCORE_NAMES)-3, len(self.methods)))

        preds_allmethods_nocrowd = -np.ones((N_nocrowd, len(self.methods)))
        probs_allmethods_nocrowd = -np.ones((N_nocrowd, self.num_classes, len(self.methods)))

        # timestamp for when we started a run. Can be compared to file versions to check what was run.
        timestamp = datetime.datetime.now().strftime('started-%Y-%m-%d-%H-%M-%S')

        model = None # pass out to other functions if required

        Ndocs = np.sum(doc_start)

        annotations_all = annotations
        doc_start_all = doc_start
        text_all = text

        if active_learning:

            rerun_all = True

            batch_size = int(np.ceil(AL_batch_fraction * Ndocs))

            seed_docs = np.random.choice(Ndocs, batch_size, replace=False)
            seed_toks = np.in1d(np.cumsum(doc_start_all) - 1, seed_docs)

        for method_idx in range(len(self.methods)):
            
            print('running method: %s' % self.methods[method_idx])
            method = self.methods[method_idx]

            if active_learning:

                selected_docs = seed_docs
                selected_toks = seed_toks

                # reset to the same original sample for the next method to be tested
                annotations = annotations_all[seed_toks, :]
                doc_start = doc_start_all[seed_toks]
                text = text_all[seed_toks]
                N_withcrowd = annotations.shape[0]

            else:
                selected_docs = None

            Nseen = 0
            while Nseen < Ndocs:
                # the active learning loop -- loop until no more labels
                # Initialise the results because not all methods will fill these in
                if test_no_crowd:
                    agg_nocrowd = np.zeros(N_nocrowd)
                    probs_nocrowd = np.ones((N_nocrowd, self.num_classes))

                if  method.split('_')[0] == 'best':
                    agg, probs = self._run_best_worker(annotations, ground_truth, doc_start)

                elif  method.split('_')[0] == 'worst':
                    agg, probs = self._run_worst_worker(annotations, ground_truth, doc_start)

                elif  method.split('_')[0] == 'majority':
                    agg, probs = majority_voting.MajorityVoting(annotations, self.num_classes).vote()

                elif  method.split('_')[0] == 'clustering':
                    agg, probs = self._run_clustering(annotations, doc_start, ground_truth)

                elif  method.split('_')[0] == 'mace':
                    tmp_path = outputdir + '/'
                    anno_path =  tmp_path + 'annos_tmp_%s' % timestamp

                    np.savetxt(anno_path, annotations, delimiter=',')

                    agg, probs = self._run_mace(anno_path, tmp_path, ground_truth)

                    os.remove(anno_path) # clean up tmp file

                elif  method.split('_')[0] == 'ds':
                    agg, probs, model = self._run_ibcc(annotations, use_ml=True)

                elif  method.split('_')[0] == 'ibcc':
                    agg, probs, model = self._run_ibcc(annotations)

                elif method.split('_')[0] == 'bac':

                    method_bits = method.split('_')

                    if method_bits[-1] == 'noHMM':
                        trans_model = 'None'
                    else:
                        trans_model = 'HMM'

                    # needs to run integrate method for task 2 as well
                    if len(method_bits) > 2 and 'integrateLSTM' in method_bits:

                        if len(method_bits) > 3 and 'atEnd' in method_bits:
                            use_LSTM = 2
                        else:
                            use_LSTM = 1
                    else:
                        use_LSTM = 0

                    if len(method_bits) > 2 and 'integrateBOF' in method_bits:

                        use_BOF = True
                    else:
                        use_BOF = False

                    if method not in self.aggs or rerun_all:
                        agg, probs, model, agg_nocrowd, probs_nocrowd = self._run_bac(annotations, doc_start, text,
                                    method, use_LSTM=use_LSTM, use_BOF=use_BOF,
                                    ground_truth_val=ground_truth_val, doc_start_val=doc_start_val, text_val=text_val,
                                    ground_truth_nocrowd=ground_truth_nocrowd, doc_start_nocrowd=doc_start_nocrowd,
                                    text_nocrowd=text_nocrowd, transition_model=trans_model)
                        self.aggs[method] = agg
                        self.probs[method] = probs
                    else:
                        agg = self.aggs[method]
                        probs = self.probs[method]

                elif 'HMM_crowd' in method:
                    if 'HMM_crowd' not in self.aggs or rerun_all:
                        # we pass all annotations here so they can be saved and reloaded from a single file in HMMCrowd
                        # format, then the relevant subset selected from that.
                        agg, probs, model = self._run_hmmcrowd(annotations_all, text_all, doc_start_all, outputdir,
                                                               selected_docs, overwrite_data_file=new_data)
                        self.aggs['HMM_crowd'] = agg
                        self.probs['HMM_crowd'] = probs
                    else:
                        agg = self.aggs['HMM_crowd']
                        probs = self.probs['HMM_crowd']

                elif 'gt' in method:
                    # for debugging
                    agg = ground_truth.flatten()
                    probs = np.zeros((len(ground_truth), self.num_classes))
                    probs[agg.astype(int)] = 1.0

                if '_then_LSTM' in method:
                    agg2, probs2, agg_nocrowd, probs_nocrowd = self._run_LSTM(N_withcrowd, text, doc_start, agg,
                                test_no_crowd, N_nocrowd, text_nocrowd, doc_start_nocrowd, ground_truth_nocrowd,
                                text_val, doc_start_val, ground_truth_val)

                    if not active_learning:
                        agg = agg2
                        probs = probs2

                if np.any(ground_truth != -1): # don't run this in the case that crowd-labelled data has no gold labels

                    if active_learning and len(agg) < len(ground_truth):
                        agg_all = np.ones(len(ground_truth))
                        agg_all[selected_toks] = agg.flatten()
                        agg = agg_all

                        probs_all = np.zeros((len(ground_truth), self.num_classes))
                        probs_all[selected_toks, :] = probs
                        probs = probs_all

                    scores_allmethods[:,method_idx][:,None], \
                    score_std_allmethods[:, method_idx] = self.calculate_scores(agg, ground_truth.flatten(), probs,
                                                                                    doc_start_all, bootstrapping)
                    preds_allmethods[:, method_idx] = agg.flatten()
                    probs_allmethods[:,:,method_idx] = probs

                if test_no_crowd:
                    scores_nocrowd[:,method_idx][:,None], \
                    score_std_nocrowd[:, method_idx] = self.calculate_scores(agg_nocrowd,
                                                        ground_truth_nocrowd.flatten(), probs_nocrowd,
                                                        doc_start_nocrowd, bootstrapping)
                    preds_allmethods_nocrowd[:, method_idx] = agg_nocrowd.flatten()
                    probs_allmethods_nocrowd[:,:,method_idx] = probs_nocrowd

                print('...done')

                # Save the results so far after each method has completed.
                Nseen = np.sum(doc_start) # update the number of documents processed so far

                # change the timestamps to include AL loop numbers
                file_identifier = timestamp + ('-Nseen%i' % Nseen)

                if outputdir is not None:
                    np.savetxt(outputdir + 'result_%s.csv' % file_identifier, scores_allmethods, fmt='%s', delimiter=',',
                               header=str(self.methods).strip('[]'))
                    np.savetxt(outputdir + 'result_std_%s.csv' % file_identifier, score_std_allmethods, fmt='%s',
                               delimiter=',',
                               header=str(self.methods).strip('[]'))

                    np.savetxt(outputdir + 'pred_%s.csv' % file_identifier, preds_allmethods, fmt='%s',
                               delimiter=',',
                               header=str(self.methods).strip('[]'))
                    with open(outputdir + 'probs_%s.pkl' % file_identifier, 'wb') as fh:
                        pickle.dump(probs_allmethods, fh)

                    np.savetxt(outputdir + 'result_nocrowd_%s.csv' % file_identifier, scores_nocrowd, fmt='%s',
                               delimiter=',',
                               header=str(self.methods).strip('[]'))
                    np.savetxt(outputdir + 'result_std_nocrowd_%s.csv' % file_identifier, score_std_nocrowd, fmt='%s',
                               delimiter=',',
                               header=str(self.methods).strip('[]'))

                    np.savetxt(outputdir + 'pred_nocrowd_%s.csv' % file_identifier, preds_allmethods_nocrowd, fmt='%s',
                               delimiter=',',
                               header=str(self.methods).strip('[]'))
                    with open(outputdir + 'probs_nocrowd_%s.pkl' % file_identifier, 'wb') as fh:
                        pickle.dump(probs_allmethods_nocrowd, fh)

                    if model is not None and not active_learning and return_model:
                        with open(outputdir + 'model_%s.pkl' % method, 'wb') as fh:
                            pickle.dump(model, fh)

                if active_learning:
                    annotations, doc_start, text, selected_docs, selected_toks = self._uncertainty_sampling(
                            annotations_all,
                            doc_start_all,
                            text_all,
                            batch_size,
                            probs,
                            selected_docs,
                            Ndocs
                        )

                    N_withcrowd = annotations.shape[0]

                    print('**** Active learning: No. annotations = %i' % N_withcrowd)
                else:
                    selected_docs = None

        if test_no_crowd:
            if return_model:
                return scores_allmethods, preds_allmethods, probs_allmethods, model, scores_nocrowd, \
                       preds_allmethods_nocrowd, probs_allmethods_nocrowd
            else:
                return scores_allmethods, preds_allmethods, probs_allmethods, scores_nocrowd, \
                       preds_allmethods_nocrowd, probs_allmethods_nocrowd
        else:
            if return_model:
                return scores_allmethods, preds_allmethods, probs_allmethods, model, None, None, None
            else:
                return scores_allmethods, preds_allmethods, probs_allmethods, None, None, None

    def data_to_hmm_crowd_format(self, annotations, text, doc_start, outputdir, docsubsetidxs, overwrite=False):

        filename = outputdir + '/hmm_crowd_text_data.pkl'

        if not os.path.exists(filename) or overwrite:
            if text is not None:

                ufeats = {}
                features = np.zeros(len(text))
                for t, tok in enumerate(text.flatten()):

                    if np.mod(t, 10000) == 0:
                        print('converting data to hmm format: token %i / %i' % (t, len(text)))

                    if tok not in ufeats:
                        ufeats[tok] = len(ufeats)

                    features[t] = ufeats[tok]
            else:
                features = []
                ufeats = []

            features = features.astype(int)

            sentences_inst = []
            crowd_labels = []

            for i in range(annotations.shape[0]):

                token_feature_vector = [features[i]]
                label = 0

                if doc_start[i]:
                    sentence_inst = []
                    sentence_inst.append(instance(token_feature_vector, label + 1))
                    sentences_inst.append(sentence_inst)

                    crowd_labs = []
                    for worker in range(annotations.shape[1]):

                        if annotations[i, worker] == -1: # worker has not seen this doc
                            continue

                        worker_labs = crowdlab(worker, len(sentences_inst) - 1, [int(annotations[i, worker])])

                        crowd_labs.append(worker_labs)
                    crowd_labels.append(crowd_labs)

                else:
                    sentence_inst.append(instance(token_feature_vector, label + 1))


                    for widx, worker_labs in enumerate(crowd_labs):
                        crowd_labs[widx].sen.append(int(annotations[i, worker_labs.wid]))

            nfeats = len(ufeats)

            print('writing HMM pickle file...')
            with open(filename, 'wb') as fh:
                pickle.dump([sentences_inst, crowd_labels, nfeats], fh)
        else:
            with open(filename, 'rb') as fh:
                data = pickle.load(fh)
            sentences_inst = data[0]
            crowd_labels = data[1]
            nfeats = data[2]

        if docsubsetidxs is not None:
            sentences_inst = np.array(sentences_inst)
            sentences_inst = sentences_inst[docsubsetidxs]

            crowd_labels = np.array(crowd_labels)
            crowd_labels = crowd_labels[docsubsetidxs]

        if isinstance(sentences_inst[0][0].features[0], float):
            # the features need to be ints not floats!
            for s, sen in enumerate(sentences_inst):
                for t, tok in enumerate(sen):
                    feat_vec = []
                    for f in range(len(tok.features)):
                        feat_vec.append(int(sentences_inst[s][t] .features[f]))
                    sentences_inst[s][t] = instance(feat_vec, sentences_inst[s][t].label)

        return sentences_inst, crowd_labels, nfeats

    def calculate_sample_metrics(self, agg, gt, probs, doc_starts):

        result = -np.ones(len(self.SCORE_NAMES)-3)

        result[0] = skm.accuracy_score(gt, agg)
        result[1] = skm.precision_score(gt, agg, average='macro')
        result[2] = skm.recall_score(gt, agg, average='macro')
        result[3] = skm.f1_score(gt, agg, average='macro')

        # token-level metrics - strict
        result[6] = precision(agg, gt, True, doc_starts)
        result[7] = recall(agg, gt, True, doc_starts)
        result[8] = f1(agg, gt, True, doc_starts)

        # token-level metrics -- relaxed
        result[9] = precision(agg, gt, False, doc_starts)
        result[10] = recall(agg, gt, False, doc_starts)
        result[11] = f1(agg, gt, False, doc_starts)

        auc_score = 0
        total_weights = 0
        for i in range(probs.shape[1]):

            if not np.any(gt == i) or np.all(gt == i):
                print('Could not evaluate AUC for class %i -- all data points have same value.' % i)
                continue

            auc_i = skm.roc_auc_score(gt == i, probs[:, i])
            # print 'AUC for class %i: %f' % (i, auc_i)
            auc_score += auc_i * np.sum(gt == i)
            total_weights += np.sum(gt == i)
        result[4] = auc_score / float(total_weights)

        result[5] = skm.log_loss(gt, probs, eps=1e-100, labels=np.arange(self.num_classes))

        return result

    def calculate_scores(self, agg, gt, probs, doc_start, bootstrapping=True):
        
        result = -np.ones((len(self.SCORE_NAMES), 1))

        # exclude data points with missing ground truth
        agg = agg[gt!=-1]
        probs = probs[gt!=-1]
        doc_start = doc_start[gt!=-1]
        gt = gt[gt!=-1]

        print('unique ground truth values: ')
        print(np.unique(gt))
        print('unique prediction values: ')
        print(np.unique(agg))
                        
        if self.postprocess:
            agg = data_utils.postprocess(agg, doc_start)
        

        print('Plotting confusion matrix for errors: ')
        
        nclasses = probs.shape[1]
        conf = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                conf[i, j] = np.sum((gt==i).flatten() & (agg==j).flatten())
                
            #print 'Acc for class %i: %f' % (i, skm.accuracy_score(gt==i, agg==i))
        print(np.round(conf))

        gold_doc_start = np.copy(doc_start)
        gold_doc_start[gt == -1] = 0

        Ndocs = int(np.sum(gold_doc_start))
        if Ndocs < 100 and bootstrapping:

            print('Using bootstrapping with small test set size')

            # use bootstrap resampling with small datasets
            nbootstraps = 100
            nmetrics = len(self.SCORE_NAMES) - 3
            resample_results = np.zeros((nmetrics, nbootstraps))

            for i in range(nbootstraps):
                print('Bootstrapping the evaluation: %i of %i' % (i, nbootstraps))
                not_sampled = True
                nsample_attempts = 0
                while not_sampled:
                    sampleidxs = np.random.choice(Ndocs, Ndocs, replace=True)
                    sampleidxs = np.in1d(np.cumsum(gold_doc_start) - 1, sampleidxs)

                    if len(np.unique(gt[sampleidxs])) >= 2:
                        not_sampled = False
                        # if we have at least two class labels in the sample ground truth, we can use this sample

                    nsample_attempts += 1
                    if nsample_attempts >= Ndocs:
                        not_sampled = False

                if nsample_attempts <= Ndocs:
                    resample_results[:, i] = self.calculate_sample_metrics(agg[sampleidxs],
                                                                       gt[sampleidxs],
                                                                       probs[sampleidxs],
                                                                       doc_start[sampleidxs])
                else: # can't find enough valid samples for the bootstrap, let's just use what we got so far.
                    resample_results = resample_results[:, :i]
                    break

            sample_res = np.mean(resample_results, axis=1)
            result[:len(sample_res), 0] = sample_res
            std_result = np.std(resample_results, axis=1)

        else:
            sample_res = self.calculate_sample_metrics(agg, gt, probs, doc_start)
            result[:len(sample_res), 0] = sample_res

            std_result = None

        result[len(sample_res)] = metrics.count_error(agg, gt)
        result[len(sample_res) + 1] = metrics.num_invalid_labels(agg, doc_start)
        result[len(sample_res) + 2] = metrics.mean_length_error(agg, gt, doc_start)

        print('F1 score tokens = %f' % result[3])
        print('F1 score spans strict = %f' % result[8])
        print('F1 score spans relaxed = %f' % result[11])

        return result, std_result
        
    def create_experiment_data(self):
        for param_idx in range(self.param_values.shape[0]):
            # update tested parameter
            if self.param_idx == 0:
                self.acc_bias = self.param_values[param_idx]
            elif self.param_idx == 1:
                self.miss_bias = self.param_values[param_idx]
            elif self.param_idx == 2:
                self.short_bias = self.param_values[param_idx]
            elif self.param_idx == 3:
                self.num_docs = self.param_values[param_idx]
            elif self.param_idx == 4:
                self.doc_length = self.param_values[param_idx]
            elif self.param_idx == 5:
                self.group_sizes = self.param_values[param_idx]
            else:
                print('Encountered invalid test index!')
                
            param_path = self.output_dir + 'data/param' + str(param_idx) + '/'
        
            for i in range(self.num_runs):
                path = param_path + 'set' + str(i) + '/'
                self.generator.init_crowd_models(self.acc_bias, self.miss_bias, self.short_bias, self.group_sizes)
                self.generator.generate_dataset(num_docs=self.num_docs, doc_length=self.doc_length, group_sizes=self.group_sizes, save_to_file=True, output_dir=path)
    
    def run_exp(self):
        # initialise result array
        results = np.zeros((self.param_values.shape[0], len(self.SCORE_NAMES), len(self.methods), self.num_runs))
        # read experiment directory
        param_dirs = glob.glob(os.path.join(self.output_dir, "data/*"))
       
        # iterate through parameter settings
        for param_idx in range(len(param_dirs)):
            
            print('parameter setting: {0}'.format(param_idx))
            
            # read parameter directory 
            path_pattern = self.output_dir + 'data/param{0}/set{1}/'

            # iterate through data sets
            for run_idx in range(self.num_runs):
                print('data set number: {0}'.format(run_idx))
                
                data_path = path_pattern.format(*(param_idx,run_idx))
                # read data
                doc_start, gt, annos = self.generator.read_data_file(data_path + 'full_data.csv')                
                # run methods
                results[param_idx,:,:,run_idx], preds, probabilities = self.run_methods(annos, gt, doc_start, data_path + 'annotations.csv')
                # save predictions
                np.savetxt(data_path + 'predictions.csv', preds)
                # save probabilities
                probabilities.dump(data_path + 'probabilities')
                
        if self.save_results:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
            print('Saving results...')
            #np.savetxt(self.output_dir + 'results.csv', np.mean(results, 3)[:, :, 0])
            results.dump(self.output_dir + 'results')

        if self.show_plots or self.save_plots:
            self.plot_results(results, self.show_plots, self.save_plots, self.output_dir)
        
        return results

    def replot_results(self):
        # Reloads predictions and probabilities from file, then computes metrics and plots.


        # initialise result array
        results = np.zeros((self.param_values.shape[0], len(self.SCORE_NAMES), len(self.methods), self.num_runs))
        std_results = np.zeros((self.param_values.shape[0], len(self.SCORE_NAMES)-3, len(self.methods), self.num_runs))
        # read experiment directory
        param_dirs = glob.glob(os.path.join(self.output_dir, "data/*"))

        # iterate through parameter settings
        for param_idx in range(len(param_dirs)):

            print('parameter setting: {0}'.format(param_idx))

            # read parameter directory
            path_pattern = self.output_dir + 'data/param{0}/set{1}/'

            # iterate through data sets
            for run_idx in range(self.num_runs):
                print('data set number: {0}'.format(run_idx))

                data_path = path_pattern.format(*(param_idx, run_idx))
                # read data
                doc_start, gt, annos = self.generator.read_data_file(data_path + 'full_data.csv')
                # run methods
                preds = np.loadtxt(data_path + 'predictions.csv')
                if preds.ndim == 1:
                    preds = preds[:, None]

                probabilities = np.load(data_path + 'probabilities')
                if probabilities.ndim == 2:
                    probabilities = probabilities[:, :, None]

                for method_idx in range(len(self.methods)):
                    print('running method: {0}'.format(self.methods[method_idx]), end=' ')

                    agg = preds[:, method_idx]
                    probs = probabilities[:, :, method_idx]

                    results[param_idx, :, method_idx, run_idx][:, None], \
                    std_results[param_idx, :, method_idx, run_idx] = self.calculate_scores(
                        agg, gt.flatten(), probs, doc_start)

        if self.save_results:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            print('Saving results...')
            # np.savetxt(self.output_dir + 'results.csv', np.mean(results, 3)[:, :, 0])
            results.dump(self.output_dir + 'results')

        if self.show_plots or self.save_plots:
            self.plot_results(results, self.show_plots, self.save_plots, self.output_dir)

        return results
    
    def run(self):
        
        if self.generate_data:
            self.create_experiment_data()
        
        if not glob.glob(os.path.join(self.output_dir, 'data/*')):
            print('No data found at specified path! Exiting!')
            return
        else:
            return self.run_exp()
        
        
    def make_plot(self, x_vals, y_vals, x_ticks_labels, ylabel):
        
        styles = ['-', '--', '-.', ':']
        markers = ['o', 'v', 's', 'p', '*']
        
        for j in range(len(self.methods)):
            plt.plot(x_vals, np.mean(y_vals[:, j, :], 1), label=self.methods[j], ls=styles[j%4], marker=markers[j%5])
                  
        plt.legend(loc='best')
        
        plt.title('parameter influence')
        plt.ylabel(ylabel)
        plt.xlabel(self.PARAM_NAMES[self.param_idx])
        plt.xticks(x_vals, x_ticks_labels)
        plt.grid(True)
        plt.grid(True, which='Minor', axis='y')
        
        if np.min(y_vals) < 0:
            plt.ylim([np.min(y_vals), np.max([1, np.max(y_vals)])])
        else:
            plt.ylim([0, np.max([1, np.max(y_vals)])])
    
    
    def plot_results(self, results, show_plot=False, save_plot=False, output_dir='/output/', score_names=None):
        
        if score_names is None: 
            score_names = self.SCORE_NAMES
        
        # create output directory if necessary
        if save_plot and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # initialise values for x axis
        if (self.param_values.ndim > 1):
            x_vals = self.param_values[:, 0]
        else:
            x_vals = self.param_values
            
        # initialise x-tick labels 
        x_ticks_labels = list(map(str, self.param_values))
            
        for i in range(len(score_names)):  
            self.make_plot(x_vals, results[:,i,:,:], x_ticks_labels, score_names[i])
        
            if save_plot:
                print('Saving plot...')
                plt.savefig(output_dir + 'plot_' + score_names[i] + '.png') 
                plt.clf()
        
            if show_plot:
                plt.show()
                
            if i == 5:
                self.make_plot(x_vals, results[:,i,:,:], x_ticks_labels, score_names[i])
                plt.ylim([0,1])
                
                if save_plot:
                    print('Saving plot...')
                    plt.savefig(output_dir + 'plot_' + score_names[i] + '_zoomed' + '.png') 
                    plt.clf()
        
                if show_plot:
                    plt.show()
