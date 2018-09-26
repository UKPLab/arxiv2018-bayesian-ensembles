'''
Created on Nov 1, 2016

@author: Melvin Laux
'''

import logging
import pickle
import pandas as pd
import datetime
import configparser
import os, subprocess
import sklearn.metrics as skm
import numpy as np
import glob

import lample_lstm_tagger.lstm_wrapper as lstm_wrapper
import evaluation.metrics as metrics
from evaluation.plots import SCORE_NAMES, plot_results
from baselines.hmm import HMM_crowd
from baselines.util import crowd_data, crowdlab, instance
from baselines import ibcc, clustering, majority_voting
from algorithm import bsc
from data import data_utils
from evaluation.span_level_f1 import precision, recall, f1, strict_span_metrics_2

logging.basicConfig(level=logging.DEBUG)

def calculate_sample_metrics(nclasses, agg, gt, probs, doc_starts, print_per_class_results=False):
    result = -np.ones(len(SCORE_NAMES) - 3)

    gt = gt.astype(int)

    probs[np.isnan(probs)] = 0
    probs[np.isinf(probs)] = 0

    # token-level metrics
    result[0] = skm.accuracy_score(gt, agg)

    # the results are undefined if some classes are not present in the gold labels
    prec_by_class = skm.precision_score(gt, agg, average=None, labels=range(nclasses))
    rec_by_class = skm.recall_score(gt, agg, average=None, labels=range(nclasses))
    f1_by_class = skm.f1_score(gt, agg, average=None, labels=range(nclasses))

    if print_per_class_results:
        print('Token Precision:')
        print(prec_by_class)

        print('Token Recall:')
        print(rec_by_class)

        print('Token F1:')
        print(f1_by_class)

    result[1] = np.mean(prec_by_class[np.unique(gt)])
    result[2] = np.mean(rec_by_class[np.unique(gt)])
    result[3] = np.mean(f1_by_class[np.unique(gt)])

    # span-level metrics - strict
    p, r, f = strict_span_metrics_2(agg, gt, doc_starts)
    result[6] = p  # precision(agg, gt, True, doc_starts)
    result[7] = r  # recall(agg, gt, True, doc_starts)
    result[8] = f  # f1(agg, gt, True, doc_starts)

    # span-level metrics -- relaxed
    result[9] = precision(agg, gt, False, doc_starts)
    result[10] = recall(agg, gt, False, doc_starts)
    result[11] = f1(agg, gt, False, doc_starts)

    auc_score = 0
    total_weights = 0
    for i in range(probs.shape[1]):

        if not np.any(gt == i) or np.all(gt == i) or np.any(np.isnan(probs[:, i])) or np.any(np.isinf(probs[:, i])):
            print('Could not evaluate AUC for class %i -- all data points have same value.' % i)
            continue

        auc_i = skm.roc_auc_score(gt == i, probs[:, i])
        # print 'AUC for class %i: %f' % (i, auc_i)
        auc_score += auc_i * np.sum(gt == i)
        total_weights += np.sum(gt == i)

        if print_per_class_results:
            print('AUC for class %i = %f' % (i, auc_i))

    result[4] = auc_score / float(total_weights) if total_weights > 0 else 0

    result[5] = skm.log_loss(gt, probs, eps=1e-100, labels=np.arange(nclasses))

    return result

def calculate_scores(nclasses, postprocess, agg, gt, probs, doc_start, bootstrapping=True, print_per_class_results=False):
    result = -np.ones((len(SCORE_NAMES), 1))

    # exclude data points with missing ground truth
    agg = agg[gt != -1]
    probs = probs[gt != -1]
    doc_start = doc_start[gt != -1]
    gt = gt[gt != -1]

    print('unique ground truth values: ')
    print(np.unique(gt))
    print('unique prediction values: ')
    print(np.unique(agg))

    if postprocess:
        agg = data_utils.postprocess(agg, doc_start)

    print('Plotting confusion matrix for errors: ')

    nclasses = probs.shape[1]
    conf = np.zeros((nclasses, nclasses))
    for i in range(nclasses):
        for j in range(nclasses):
            conf[i, j] = np.sum((gt == i).flatten() & (agg == j).flatten())

        # print 'Acc for class %i: %f' % (i, skm.accuracy_score(gt==i, agg==i))
    print(np.round(conf))

    gold_doc_start = np.copy(doc_start)
    gold_doc_start[gt == -1] = 0

    Ndocs = int(np.sum(gold_doc_start))
    if Ndocs < 100 and bootstrapping:

        print('Using bootstrapping with small test set size')

        # use bootstrap resampling with small datasets
        nbootstraps = 100
        nmetrics = len(SCORE_NAMES) - 3
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
                resample_results[:, i] = calculate_sample_metrics(nclasses,
                                                                  agg[sampleidxs],
                                                                  gt[sampleidxs],
                                                                  probs[sampleidxs],
                                                                  doc_start[sampleidxs],
                                                                  print_per_class_results)
            else:  # can't find enough valid samples for the bootstrap, let's just use what we got so far.
                resample_results = resample_results[:, :i]
                break

        sample_res = np.mean(resample_results, axis=1)
        result[:len(sample_res), 0] = sample_res
        std_result = np.std(resample_results, axis=1)

    else:
        sample_res = calculate_sample_metrics(nclasses, agg, gt, probs, doc_start, print_per_class_results)
        result[:len(sample_res), 0] = sample_res

        std_result = None

    result[len(sample_res)] = metrics.count_error(agg, gt)
    result[len(sample_res) + 1] = metrics.num_invalid_labels(agg, doc_start)
    result[len(sample_res) + 2] = metrics.mean_length_error(agg, gt, doc_start)

    print('F1 score tokens = %f' % result[3])
    print('F1 score spans strict = %f' % result[8])
    print('F1 score spans relaxed = %f' % result[11])

    return result, std_result

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

    save_results = False
    save_plots = False
    show_plots = False

    postprocess = False # previous papers did not use this so we leave out to make results comparable.
    # Only simple IOB2 is implemented so far.

    opt_hyper = False

    crf_probs = False

    random_sampling = False

    def __init__(self, generator, nclasses=None, nannotators=None, config=None,
                 alpha0_factor=1.0, alpha0_diags=1.0, beta0_factor=0.1, max_iter=20, crf_probs=False, rep=0):

        self.output_dir = '~/data/bayesian_annotator_combination/output/'

        self.crf_probs = crf_probs

        self.generator = generator

        if not (config == None):
            self.config_file = config
            self.read_config_file()

        else:
            self.num_classes = nclasses
            self.nannotators = nannotators

        self.output_dir = os.path.expanduser(self.output_dir)

        self.alpha0_factor = alpha0_factor
        self.alpha0_diags = alpha0_diags
        self.nu0_factor = beta0_factor

        # save results from methods here. If we use compound methods, we can reuse these results in different
        # combinations of methods.
        self.aggs = {}
        self.probs = {}

        self.max_iter = max_iter # allow all methods to use a maximum no. iterations

        np.random.seed(3849)
        self.seed = np.random.randint(1, 1000, 100)[rep] # seeds for AL

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

    def tune_alpha0(self, alpha0diag_proposals, alpha0factor_proposals, nu0factor_proposals, method,
                    annotations, ground_truth, doc_start,
                    outputdir, text, tune_lstm=False, metric_idx_to_optimise=8,
                    ground_truth_val=None, doc_start_val=None, text_val=None):

        self.methods = [method]

        scores = np.zeros((len(nu0factor_proposals) * len(alpha0diag_proposals), len(alpha0factor_proposals)))

        best_scores = np.zeros(4)
        best_idxs = np.zeros(4)

        for h, nu0factor in enumerate(nu0factor_proposals):

            self.nu0_factor = nu0factor

            for i, alpha0diag in enumerate(alpha0diag_proposals):

                if tune_lstm:
                    self.alpha0_diags_lstm = alpha0diag
                else:
                    self.alpha0_diags = alpha0diag

                for j, alpha0factor in enumerate(alpha0factor_proposals):

                    # reset saved data so that models are run again.
                    self.aggs = {}
                    self.probs = {}

                    outputdir_ij = outputdir + ('_%i_%i_%i_' % (h, i, j)) + method + '/'

                    if tune_lstm:
                        self.alpha0_factor_lstm = alpha0factor
                    else:
                        self.alpha0_factor = alpha0factor

                    all_scores, _, _, _, _, _ = self.run_methods(annotations, ground_truth, doc_start,
                                                                 outputdir_ij,
                                                                 text,
                                                                 ground_truth_val=ground_truth_val,
                                                                 doc_start_val=doc_start_val,
                                                                 text_val=text_val,
                                                                 bootstrapping=False)
                    scores[(h*len(alpha0diag_proposals)) + i, j] = all_scores[metric_idx_to_optimise, :] # 3 is F1score
                    print('Scores for %f, %f, %f: %f' % (nu0factor, alpha0diag, alpha0factor,
                                                         scores[(h*len(alpha0diag_proposals)) + i, j]))

                    if scores[(h*len(alpha0diag_proposals)) + i, j] > best_scores[0]:
                        best_scores[0] = scores[(h*len(alpha0diag_proposals)) + i, j]
                        best_scores[1] = nu0factor
                        best_scores[2] = alpha0diag
                        best_scores[3] = alpha0factor

                        best_idxs[0] = scores[(h*len(alpha0diag_proposals)) + i, j]
                        best_idxs[1] = h
                        best_idxs[2] = i
                        best_idxs[3] = j

                    print('Saving scores for this setting to %s' % (outputdir + '/%s_scores.csv' % method))
                    np.savetxt(outputdir + '/%s_scores.csv' % method, scores, fmt='%s', delimiter=',',
                               header=str(self.methods).strip('[]'))

                    np.savetxt(outputdir + '/%s_bestscores.csv' % method, best_scores, fmt='%s', delimiter=',',
                               header=str(self.methods).strip('[]'))

        return best_idxs

    def tune_acc_bias(self, acc_bias_proposals, method,
                    annotations, ground_truth, doc_start,
                    outputdir, text, tune_lstm=False, metric_idx_to_optimise=8,
                    ground_truth_val=None, doc_start_val=None, text_val=None):

        self.methods = [method]

        scores = np.zeros(len(acc_bias_proposals))

        best_score = -np.inf
        best_idx = 0

        for i, acc_bias in enumerate(acc_bias_proposals):

            self.alpha0_acc_bias = acc_bias

            # reset saved data so that models are run again.
            self.aggs = {}
            self.probs = {}

            outputdir_i = outputdir + ('_acc_bias_%i' % (i)) + method + '/'

            all_scores, _, _, _, _, _ = self.run_methods(annotations, ground_truth, doc_start,
                                                                 outputdir_i,
                                                                 text,
                                                                 ground_truth_val=ground_truth_val,
                                                                 doc_start_val=doc_start_val,
                                                                 text_val=text_val,
                                                                 bootstrapping=False)
            scores[i] = all_scores[metric_idx_to_optimise, :] # 3 is F1score

            print('Scores for %f: %f' % (acc_bias, scores[i]))

            if scores[i] > best_score:
                best_score = scores[i]
                best_idx = scores[i]

                print('Saving scores for this setting to %s' % (outputdir + '/%s_scores.csv' % method))
                np.savetxt(outputdir + '/%s_acc_bias_scores.csv' % method, scores, fmt='%s', delimiter=',',
                               header=str(self.methods).strip('[]'))

        return best_score, best_idx

    def _run_best_worker(self, annos, gt, doc_start):
        # choose the best classifier by f1-score
        f1scores = np.zeros_like(annos) - 1.0
        print('F1 scores for individual workers:')

        individual_scores = []

        for w in range(annos.shape[1]):

            valididxs = annos[:, w] != -1

            if not np.any(valididxs):
                continue

            f1_by_class = skm.f1_score(gt.flatten()[valididxs], annos[valididxs, w], average=None)
            f1_w = np.mean(f1_by_class[np.unique(gt[valididxs]).astype(int)])

            #print(f1_w)
            individual_scores.append(f1_w)

            f1scores[valididxs, w] = f1_w

        #print(sorted(individual_scores))

        best_idxs = np.argmax(f1scores, axis=1)
        agg = annos[np.arange(annos.shape[0]), best_idxs]
        probs = np.zeros((gt.shape[0], self.num_classes))
        for k in range(gt.shape[0]):
            probs[k, int(agg[k])] = 1

        return agg, probs

    def _run_worst_worker(self, annos, gt, doc_start):
        # choose the weakest classifier by f1-score
        f1scores = np.zeros_like(annos) + np.inf
        for w in range(annos.shape[1]):

            valididxs = annos[:, w] != -1

            if not np.any(valididxs):
                continue

            f1_by_class = skm.f1_score(gt.flatten()[valididxs], annos[valididxs, w], average=None)
            f1scores[valididxs, w] = np.mean(f1_by_class[np.unique(gt[valididxs]).astype(int)])

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

    def _run_mace(self, anno_path, tmp_path, ground_truth, annotations):

        annotations = pd.DataFrame(annotations)
        annotations.replace(-1, np.nan)
        annotations.to_csv(anno_path, sep=',', header=False, index=False)

        #np.savetxt(anno_path, annotations, delimiter=',')

        subprocess.call(['java', '-jar', './MACE/MACE.jar', '--distribution', '--prefix',
                        tmp_path + '/mace',
                        anno_path])  # , stdout = devnull, stderr = devnull)

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
        ibc.max_iterations = self.max_iter
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
                 ground_truth_nocrowd=None, doc_start_nocrowd=None, text_nocrowd=None, transition_model='HMM',
                 doc_start_unseen=None, text_unseen=None):

        self.bac_worker_model = method.split('_')[1]
        L = self.num_classes

        num_types = (self.num_classes - 1) / 2
        outside_labels = [-1, 1]
        inside_labels = (np.arange(num_types) * 2 + 1).astype(int)
        inside_labels[0] = 0
        begin_labels = (np.arange(num_types) * 2 + 2).astype(int)

        data_model = []

        if use_LSTM > 0:
            data_model.append('LSTM')
        if use_BOF:
            data_model.append('IF')

        bac_model = bsc.BAC(L=L, K=annotations.shape[1], max_iter=self.max_iter,
                    inside_labels=inside_labels, outside_labels=outside_labels, beginning_labels=begin_labels,
                    alpha0_diags=self.alpha0_diags, alpha0_factor=self.alpha0_factor,
                    nu0_factor=self.nu0_factor,
                    exclusions=self.exclusions, before_doc_idx=1, worker_model=self.bac_worker_model,
                    tagging_scheme='IOB2', data_model=data_model, transition_model=transition_model)

        bac_model.verbose = True

        if self.opt_hyper:
            np.random.seed(592)  # for reproducibility
            probs, agg = bac_model.optimize(annotations, doc_start, text, maxfun=1000,
                                      converge_workers_first=use_LSTM==2)
        else:
            probs, agg = bac_model.run(annotations, doc_start, text,
                                 converge_workers_first=use_LSTM==2, crf_probs=self.crf_probs)

        model = bac_model

        if ground_truth_nocrowd is not None and '_thenLSTM' not in method:
            probs_nocrowd, agg_nocrowd = bac_model.predict(doc_start_nocrowd, text_nocrowd)
        else:
            probs_nocrowd = None
            agg_nocrowd = None

        if '_thenLSTM' not in method and doc_start_unseen is not None and len(doc_start_unseen) > 0:
            probs_unseen, agg_unseen = bac_model.predict(doc_start_unseen, text_unseen)
        else:
            probs_unseen = None
            agg_unseen = None

        return agg, probs, model, agg_nocrowd, probs_nocrowd, agg_unseen, probs_unseen

    def _run_hmmcrowd(self, annotations, text, doc_start, outputdir, doc_subset, nselected_by_doc,
                      overwrite_data_file=False):

        sentences, crowd_labels, nfeats = self.data_to_hmm_crowd_format(annotations, text, doc_start, outputdir,
                                                                        doc_subset, nselected_by_doc,
                                                                        overwrite=overwrite_data_file)

        data = crowd_data(sentences, crowd_labels)
        hc = HMM_crowd(self.num_classes, nfeats, data, None, None, n_workers=annotations.shape[1],
                       vb=[0.1, 0.1], ne=1)
        hc.init(init_type='dw', wm_rep='cv2', dw_em=5, wm_smooth=0.1)

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

    def _run_LSTM(self, N_withcrowd, text, doc_start, train_labs, test_no_crowd, N_nocrowd, text_nocrowd, doc_start_nocrowd,
                  ground_truth_nocrowd, text_val, doc_start_val, ground_truth_val, timestamp,
                  doc_start_unseen=None, text_unseen=None):
        '''
        Reasons why we need the LSTM integration:
        - Could use a simpler model like hmm_crowd for task 1
        - Performance may be better with LSTM
        - Active learning possible with simpler model, but how novel?
        - Integrating LSTM shows a generalisation -- step forward over models that fix the data representation
        - Can do task 2 better than a simple model like Hmm_crowd
        - Can do task 2 better than training on HMM_crowd then LSTM, or using LSTM-crowd.s
        '''
        labelled_sentences, IOB_map, IOB_label = lstm_wrapper.data_to_lstm_format(N_withcrowd, text, doc_start,
                                                                          train_labs.flatten(), self.num_classes)
        np.random.seed(592) # for reproducibility

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

        lstm = lstm_wrapper.LSTMWrapper('./models_LSTM_%s' % timestamp)

        print('Running LSTM with crf probs = %s' % self.crf_probs)

        lstm.train_LSTM(all_sentences, train_sentences, dev_sentences, ground_truth_val, IOB_map,
                        IOB_label, self.num_classes, freq_eval=self.max_iter, n_epochs=self.max_iter,
                        crf_probs=self.crf_probs, max_niter_no_imprv=2)

        # now make predictions for all sentences
        agg, probs = lstm.predict_LSTM(labelled_sentences)

        if test_no_crowd:
            labelled_sentences, IOB_map, _ = lstm_wrapper.data_to_lstm_format(N_nocrowd, text_nocrowd,
                                                          doc_start_nocrowd, np.ones(N_nocrowd), self.num_classes)

            agg_nocrowd, probs_nocrowd = lstm.predict_LSTM(labelled_sentences)

        else:
            agg_nocrowd = None
            probs_nocrowd = None

        if doc_start_unseen is not None:
            N_unseen = len(doc_start_unseen)
            labelled_sentences, IOB_map, _ = lstm_wrapper.data_to_lstm_format(N_unseen, text_unseen,
                                                                              doc_start_unseen, np.ones(N_unseen),
                                                                              self.num_classes)

            agg_unseen, probs_unseen = lstm.predict_LSTM(labelled_sentences)

        else:
            agg_unseen = None
            probs_unseen = None

        return agg, probs, agg_nocrowd, probs_nocrowd, agg_unseen, probs_unseen

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

    def _uncertainty_sampling(self, annos_all, doc_start_all, text_all, batch_size, probs, annos_sofar, selected_toks, selected_docs):

        unfinished_toks = np.ones(annos_all.shape[0], dtype=bool)

        if annos_sofar is not None:
            unfinished_toks[selected_toks] = (np.sum(annos_all[selected_toks] != -1, axis=1) - np.sum(annos_sofar != -1, axis=1)) > 0

        unfinished_toks = np.sort(np.argwhere(unfinished_toks).flatten())

        # random selection
        #new_selection = np.random.choice(unseen_docs, batch_size, replace=False)

        probs = probs[unfinished_toks, :]
        negentropy = np.log(probs) * probs
        negentropy[probs == 0] = 0
        negentropy = np.sum(negentropy, axis=1)

        docids_by_tok = (np.cumsum(doc_start_all) - 1)[unfinished_toks]

        Ndocs = np.max(docids_by_tok) + 1

        if self.random_sampling:
            new_selected_docs = np.random.choice(np.unique(docids_by_tok), batch_size, replace=False)

        else:
            negentropy_docs = np.zeros(Ndocs, dtype=float)
            count_unseen_toks = np.zeros(Ndocs, dtype=float)

            # now sum up the entropy for each doc and normalise by length (otherwise we'll never label the short ones)
            for i, _ in enumerate(unfinished_toks):
                # find doc ID for this tok
                docid = docids_by_tok[i]

                negentropy_docs[docid] += negentropy[i]
                count_unseen_toks[docid] += 1

            negentropy_docs /= count_unseen_toks

            # assume that batch size is less than number of docs...
            new_selected_docs = np.argsort(negentropy_docs)[:batch_size]

        new_selected_toks = np.in1d(np.cumsum(doc_start_all) - 1, new_selected_docs)

        new_label_count = np.zeros(annos_all.shape[0])
        new_label_count[new_selected_toks] += 1

        # add to previously selected toks and docs
        if annos_sofar is not None:
            # how many labels do we have for these tokens so far?
            current_label_count = np.sum(annos_sofar != -1, axis=1)

            # add one for the new round
            new_label_count[selected_toks] += current_label_count

            selected_docs = np.sort(np.unique(np.concatenate((selected_docs, new_selected_docs)) ))

            new_selected_toks[selected_toks] = True
            selected_toks = new_selected_toks
        else:
            selected_toks = new_selected_toks
            selected_docs = new_selected_docs

        nselected_by_doc = new_label_count[selected_toks]
        nselected_by_doc = nselected_by_doc[doc_start_all[selected_toks].flatten() == 1]

        # find the columns for each token that we should get from the full set of crowdsource annos
        mask = np.cumsum(annos_all[selected_toks] != -1, axis=1) <= new_label_count[selected_toks][:, None]

        # make a new label set
        annotations = np.zeros((np.sum(selected_toks), annos_all.shape[1])) - 1

        # copy in the data from annos_all
        annotations[mask] = annos_all[selected_toks, :][mask]

        doc_start = doc_start_all[selected_toks]
        text = text_all[selected_toks]

        return annotations, doc_start, text, selected_docs, np.argwhere(selected_toks).flatten(), nselected_by_doc

    def run_methods(self, annotations, ground_truth, doc_start, outputdir=None, text=None,
                    ground_truth_nocrowd=None, doc_start_nocrowd=None, text_nocrowd=None,
                    ground_truth_val=None, doc_start_val=None, text_val=None,
                    return_model=False, rerun_all=False, new_data=False,
                    active_learning=False, AL_batch_fraction=0.05, max_AL_iters=10,
                    bootstrapping=True, ground_truth_all_points=None):
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

        scores_allmethods = np.zeros((len(SCORE_NAMES), len(self.methods)))
        score_std_allmethods = np.zeros((len(SCORE_NAMES)-3, len(self.methods)))

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
        scores_nocrowd = np.zeros((len(SCORE_NAMES), len(self.methods)))
        score_std_nocrowd = np.zeros((len(SCORE_NAMES)-3, len(self.methods)))

        preds_allmethods_nocrowd = -np.ones((N_nocrowd, len(self.methods)))
        probs_allmethods_nocrowd = -np.ones((N_nocrowd, self.num_classes, len(self.methods)))

        # timestamp for when we started a run. Can be compared to file versions to check what was run.
        timestamp = datetime.datetime.now().strftime('started-%Y-%m-%d-%H-%M-%S')

        model = None # pass out to other functions if required

        Ndocs = np.sum(doc_start)
        print('Size of training set = %i documents' % Ndocs)

        annotations_all = annotations
        doc_start_all = doc_start
        text_all = text

        self.bac_nu0 = np.ones((self.num_classes + 1, self.num_classes)) * self.nu0_factor
        self.ibcc_nu0 = np.ones(self.num_classes) * self.nu0_factor

        # indicates presence of annotations for each document from each annotator
        annos_indicator = annotations_all[(doc_start_all == 1).flatten(), :] != -1
        # get total annotation count
        Nannos = np.sum(annos_indicator)

        for method_idx in range(len(self.methods)):

            print('running method: %s' % self.methods[method_idx])
            method = self.methods[method_idx]

            Nseen = 0
            niter = 0

            if active_learning:

                rerun_all = True

                # get the number of labels to select each iteration
                batch_size = int(np.ceil(AL_batch_fraction * Nannos))

                np.random.seed(self.seed)  # for repeating with different methods with same initial set

                annotations, doc_start, text, selected_docs, selected_toks, nselected_by_doc = self._uncertainty_sampling(
                    annotations_all,
                    doc_start_all,
                    text_all,
                    batch_size,
                    np.ones((annotations_all.shape[0], self.num_classes), dtype=float) / self.num_classes,
                    None,
                    None,
                    None
                )

                N_withcrowd = annotations.shape[0]
            else:
                selected_docs = None
                nselected_by_doc = None

            while Nseen < Nannos and niter < max_AL_iters:

                print('Learning round %i' % niter)
                niter += 1

                # the active learning loop -- loop until no more labels
                # Initialise the results because not all methods will fill these in
                if test_no_crowd:
                    agg_nocrowd = np.zeros(N_nocrowd)
                    probs_nocrowd = np.ones((N_nocrowd, self.num_classes))

                if active_learning: # treat the unseen instances similarly to the no crowd instances -- get their posterior probabilities

                    unseen_docs = np.arange(Ndocs)
                    unseen_docs = unseen_docs[np.invert(np.in1d(unseen_docs, selected_docs))]

                    unselected_toks = np.argwhere(np.in1d(np.cumsum(doc_start_all) - 1, unseen_docs)).flatten()
                    N_unseentoks = len(unselected_toks)

                    agg_unseen = np.zeros(N_unseentoks)
                    # default maximum entropy situation
                    probs_unseen = np.ones((N_unseentoks, self.num_classes), dtype=float) / self.num_classes

                    doc_start_unseen = doc_start_all[unselected_toks]
                    text_unseen = text_all[unselected_toks]
                else:
                    unseen_docs = []
                    unselected_toks = []
                    agg_unseen = None
                    probs_unseen = None
                    annotations_unseen = None
                    doc_start_unseen = None
                    text_unseen = None

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

                    agg, probs = self._run_mace(anno_path, tmp_path, ground_truth, annotations)

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
                        agg, probs, model, agg_nocrowd, probs_nocrowd, agg_unseen, probs_unseen = self._run_bac(
                                    annotations, doc_start, text,
                                    method, use_LSTM=use_LSTM, use_BOF=use_BOF,
                                    ground_truth_val=ground_truth_val, doc_start_val=doc_start_val, text_val=text_val,
                                    ground_truth_nocrowd=ground_truth_nocrowd,
                                    doc_start_nocrowd=doc_start_nocrowd,
                                    text_nocrowd=text_nocrowd,
                                    transition_model=trans_model,
                                    doc_start_unseen=doc_start_unseen,
                                    text_unseen=text_unseen)

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
                                                               selected_docs, nselected_by_doc,
                                                               overwrite_data_file=new_data)
                        self.aggs['HMM_crowd'] = agg
                        self.probs['HMM_crowd'] = probs
                    else:
                        agg = self.aggs['HMM_crowd']
                        probs = self.probs['HMM_crowd']

                elif 'gt' in method:
                    # for debugging
                    if ground_truth_all_points is not None:
                        agg = ground_truth_all_points.flatten()
                    else:
                        agg = ground_truth.flatten()

                    probs = np.zeros((len(ground_truth), self.num_classes))
                    probs[range(len(agg)), agg.astype(int)] = 1.0

                if '_then_LSTM' in method:
                    agg, probs, agg_nocrowd, probs_nocrowd, agg_unseen, probs_unseen = self._run_LSTM(N_withcrowd, text, doc_start, agg,
                                test_no_crowd, N_nocrowd, text_nocrowd, doc_start_nocrowd, ground_truth_nocrowd,
                                text_val, doc_start_val, ground_truth_val, timestamp, doc_start_unseen, text_unseen)

                if np.any(ground_truth != -1): # don't run this in the case that crowd-labelled data has no gold labels

                    if active_learning and len(agg) < len(ground_truth):
                        agg_all = np.ones(len(ground_truth))
                        agg_all[selected_toks] = agg.flatten()

                        probs_all = np.zeros((len(ground_truth), self.num_classes))
                        probs_all[selected_toks, :] = probs

                        agg_all[unselected_toks] = agg_unseen.flatten()
                        probs_all[unselected_toks, :] = probs_unseen

                        agg = agg_all
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
                Nseen = np.sum(annotations[doc_start.flatten()==1] != -1) # update the number of documents processed so far
                print('Nseen = %i' % Nseen)

                # change the timestamps to include AL loop numbers
                file_identifier = timestamp + ('-Nseen%i' % Nseen)

                if outputdir is not None:

                    self.append_method_result_to_csv(outputdir, method_idx, scores_allmethods,
                                                     'result', file_identifier)
                    self.append_method_result_to_csv(outputdir, method_idx, score_std_allmethods,
                                                     'result_std', file_identifier)
                    self.append_method_result_to_csv(outputdir, method_idx, preds_allmethods,
                                                     'pred', file_identifier)
                    self.append_method_result_to_csv(outputdir, method_idx, scores_nocrowd,
                                                     'result_nocrowd', file_identifier)
                    self.append_method_result_to_csv(outputdir, method_idx, score_std_nocrowd,
                                                     'result_std_nocrowd', file_identifier)
                    self.append_method_result_to_csv(outputdir, method_idx, preds_allmethods_nocrowd,
                                                     'pred_nocrowd', file_identifier)

                    with open(outputdir + 'probs_%s.pkl' % file_identifier, 'wb') as fh:
                        pickle.dump(probs_allmethods, fh)

                    with open(outputdir + 'probs_nocrowd_%s.pkl' % file_identifier, 'wb') as fh:
                        pickle.dump(probs_allmethods_nocrowd, fh)

                    if model is not None and not active_learning and return_model:
                        with open(outputdir + 'model_%s.pkl' % method, 'wb') as fh:
                            pickle.dump(model, fh)

                if active_learning and Nseen < Nannos:
                    annotations, doc_start, text, selected_docs, selected_toks, nselected_by_doc = self._uncertainty_sampling(
                            annotations_all,
                            doc_start_all,
                            text_all,
                            batch_size,
                            probs,
                            annotations,
                            selected_toks,
                            selected_docs
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

    def append_method_result_to_csv(self, outputdir, method_idx, new_data, filename, file_identifier):

        filename = outputdir + '%s_%s.csv' % (filename, file_identifier)

        new_data = pd.DataFrame(new_data[:, method_idx], columns=[str(self.methods[method_idx]).strip('[]')])

        if os.path.isfile(filename):
            data = pd.read_csv(filename, delimiter=',')
            expanded_data = pd.concat((data, new_data), axis=1)
        else:
            expanded_data = new_data

        expanded_data.to_csv(filename, index=False)

    def data_to_hmm_crowd_format(self, annotations, text, doc_start, outputdir, docsubsetidxs, nselected_by_doc, overwrite=False):

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

                features = features.astype(int)

            else:
                features = []
                ufeats = []


            sentences_inst = []
            crowd_labels = []

            for i in range(annotations.shape[0]):

                if len(features):
                    token_feature_vector = [features[i]]
                else:
                    token_feature_vector = []

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

            crowd_labels = [crowd_labels[d] for d in docsubsetidxs]

            for d, crowd_labels_d in enumerate(crowd_labels):
                crowd_labels[d] = crowd_labels_d[:int(nselected_by_doc[d])]
            crowd_labels = np.array(crowd_labels)

        if len(sentences_inst[0][0].features) and isinstance(sentences_inst[0][0].features[0], float):
            # the features need to be ints not floats!
            for s, sen in enumerate(sentences_inst):
                for t, tok in enumerate(sen):
                    feat_vec = []
                    for f in range(len(tok.features)):
                        feat_vec.append(int(sentences_inst[s][t] .features[f]))
                    sentences_inst[s][t] = instance(feat_vec, sentences_inst[s][t].label)

        return sentences_inst, crowd_labels, nfeats

    def calculate_scores(self, agg, gt, probs, doc_start, bootstrapping=True):
        return calculate_scores(self.num_classes, self.postprocess, agg, gt, probs, doc_start,
                                bootstrapping)

# CODE RELATING TO SYNTHETIC DATA EXPERIMENTS --------------------------------------------------------------------------
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
                self.generator.generate_dataset(num_docs=self.num_docs, doc_length=self.doc_length,
                                                group_sizes=self.group_sizes, save_to_file=True, output_dir=path)

    def run_synth_exp(self):
        # initialise result array
        results = np.zeros((self.param_values.shape[0], len(SCORE_NAMES), len(self.methods), self.num_runs))

        # iterate through parameter settings
        for param_idx in range(self.param_values.shape[0]):

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
                results[param_idx,:,:,run_idx], preds, probabilities, _, _, _ = self.run_methods(annos, gt, doc_start,
                                                        data_path, new_data=True, bootstrapping=False, rerun_all=True)
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
            plot_results(self.param_values, self.methods, self.param_idx, results, self.show_plots,
                         self.save_plots, self.output_dir)

        return results

    def replot_results(self, exclude_methods=[], recompute=True):
        # Reloads predictions and probabilities from file, then computes metrics and plots.

        # initialise result array
        results = np.zeros((self.param_values.shape[0], len(SCORE_NAMES), len(self.methods), self.num_runs))
        std_results = np.zeros((self.param_values.shape[0], len(SCORE_NAMES)-3, len(self.methods), self.num_runs))

        if recompute:

            # iterate through parameter settings
            for param_idx in range(self.param_values.shape[0]):

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

                        if self.methods[method_idx] in exclude_methods:
                            continue

                        print('running method: {0}'.format(self.methods[method_idx]), end=' ')

                        agg = preds[:, method_idx]
                        probs = probabilities[:, :, method_idx]

                        results[param_idx, :, method_idx, run_idx][:, None], \
                        std_results[param_idx, :, method_idx, run_idx] = self.calculate_scores(
                            agg, gt.flatten(), probs, doc_start, bootstrapping=False)

        else:
            print('Loading precomputed results...')
            # np.savetxt(self.output_dir + 'results.csv', np.mean(results, 3)[:, :, 0])
            results = np.load(self.output_dir + 'results')

        included_methods = np.ones(len(self.methods), dtype=bool)
        included_methods[np.in1d(self.methods, exclude_methods)] = False
        included_methods = np.argwhere(included_methods).flatten()

        results = results[:, :, included_methods, :]

        methods = np.array(self.methods)[included_methods]

        if self.save_results:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            print('Saving results...')
            # np.savetxt(self.output_dir + 'results.csv', np.mean(results, 3)[:, :, 0])
            results.dump(self.output_dir + 'results')

        if self.show_plots or self.save_plots:
            print('making plots for parameter setting: {0}'.format(self.param_idx))
            plot_results(self.param_values, methods, self.param_idx, results,
                         self.show_plots, self.save_plots, self.output_dir)

        return results

    def run_synth(self):

        if self.generate_data:
            self.create_experiment_data()

        if not glob.glob(os.path.join(self.output_dir, 'data/*')):
            print('No data found at specified path! Exiting!')
            return
        else:
            return self.run_synth_exp()


