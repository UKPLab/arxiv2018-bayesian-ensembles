'''
Created on Nov 1, 2016

@author: Melvin Laux
'''
import time

#from scipy.optimize import minimize

import logging
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
import matplotlib.pyplot as plt
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
    
    PARAM_NAMES = ['acc_bias', 'miss_bias', 'short_bias', 'num_docs', 'doc_length', 'group_sizes']
    SCORE_NAMES = ['accuracy', 'precision', 'recall', 'f1-score', 'auc-score', 'cross-entropy-error', 'count error', 
                   'number of invalid labels', 'mean length error']
    
    generate_data= False
    
    methods = None

    alpha0_factor = 16.0
    alpha0_diags = 1.0
    nu0_factor = 100.0

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
    
    postprocess = True

    opt_hyper = False
    
    '''
    Constructor
    '''
    def __init__(self, generator, nclasses, nannotators, config=None):
        '''
        Constructor
        '''
        self.generator = generator
        
        if not (config == None):
            self.config_file = config
            self.read_config_file()

        print('setting initial priors for Bayesian methods...')

        self.num_classes = nclasses
        self.nannotators = nannotators

        self.ibcc_alpha0 = self.alpha0_factor * np.ones((nclasses, nclasses)) + self.alpha0_diags * np.eye(nclasses)

        # TODO check whether constraints on transitions are correctly encoded
        self.bac_nu0 = np.ones((nclasses + 1, nclasses)) * self.nu0_factor
        self.ibcc_nu0 = np.ones(nclasses) * self.nu0_factor
            
            
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

    def _run_mace(self, anno_path, ground_truth):
        # devnull = open(os.devnull, 'w')
        subprocess.call(['java', '-jar', './MACE/MACE.jar', '--distribution', '--prefix',
                         '../../data/bayesian_annotator_combination/output/data/mace',
                         anno_path])  # , stdout = devnull, stderr = devnull)

        result = np.genfromtxt('../../data/bayesian_annotator_combination/output/data/mace.prediction')

        agg = result[:, 0]

        probs = np.zeros((ground_truth.shape[0], self.num_classes))
        for i in range(result.shape[0]):
            for j in range(0, self.num_classes * 2, 2):
                probs[i, int(result[i, j])] = result[i, j + 1]

        return agg, probs

    def _run_ibcc(self, annotations):
        ibc = ibcc.IBCC(nclasses=self.num_classes, nscores=self.num_classes, nu0=self.ibcc_nu0,
                        alpha0=self.ibcc_alpha0, uselowerbound=True)
        # ibc.verbose = True
        # ibc.optimise_alpha0_diagonals = True

        if self.opt_hyper:
            probs = ibc.combine_classifications(annotations, table_format=True, optimise_hyperparams=True,
                                                maxiter=10000)
        else:
            probs = ibc.combine_classifications(annotations, table_format=True)  # posterior class probabilities
        agg = probs.argmax(axis=1)  # aggregated class labels

        return agg, probs

    def _run_bac(self, annotations, doc_start, text, method):
        self.bac_worker_model = method.split('_')[1]
        L = self.num_classes

        # matrices are repeated for the different annotators/previous label conditions inside the BAC code itself.
        if self.bac_worker_model == 'seq' or self.bac_worker_model == 'ibcc':
            self.bac_alpha0 = self.alpha0_factor * np.ones((L, L)) + \
                              self.alpha0_diags * np.eye(L)

        elif self.bac_worker_model == 'mace':
            self.bac_alpha0 = self.alpha0_factor * np.ones((2 + L))
            self.bac_alpha0[1] += self.alpha0_diags  # diags are bias toward correct answer

        elif self.bac_worker_model == 'acc':
            self.bac_alpha0 = self.alpha0_factor * np.ones((2))
            self.bac_alpha0[1] += self.alpha0_diags  # diags are bias toward correct answer

        if L == 7:
            inside_labels = [0, 3, 5]
            outside_labels = [-1, 1]
            begin_labels = [2, 4, 6]
        else:
            inside_labels = [0]
            outside_labels = [-1, 1]
            begin_labels = [2]

        alg = bac.BAC(L=L, K=annotations.shape[1], inside_labels=inside_labels, outside_labels=outside_labels,
                      beginning_labels=begin_labels, alpha0=self.bac_alpha0, nu0=self.bac_nu0,
                      exclusions=self.exclusions, before_doc_idx=-1, worker_model=self.bac_worker_model)
        # alg.max_iter = 1
        alg.verbose = True
        if self.opt_hyper:
            probs, agg = alg.optimize(annotations, doc_start, text, maxfun=1000)
        else:
            probs, agg = alg.run(annotations, doc_start, text)

        model = alg

        return agg, probs, model

    def run_methods(self, annotations, ground_truth, doc_start, outputdir=None, text=None, anno_path=None,
                    ground_truth_nocrowd=None, doc_start_nocrowd=None, text_nocrowd=None,
                    ground_truth_val=None, doc_start_val=None, text_val=None,
                    return_model=False):
        '''

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

        print('Running experiment, Optimisation=%s' % (self.opt_hyper))

        scores_allmethods = np.zeros((len(self.SCORE_NAMES), len(self.methods)))
        score_std_allmethods = np.zeros((len(self.SCORE_NAMES)-3, len(self.methods)))

        preds_allmethods = -np.ones((annotations.shape[0], len(self.methods)))
        probs_allmethods = -np.ones((annotations.shape[0], self.num_classes, len(self.methods)))

        # a second test set with no crowd labels was supplied directly
        if ground_truth_nocrowd is not None and text_nocrowd is not None and doc_start_nocrowd:
            test_no_crowd = True

        else:
            crowd_labels_present = np.any(annotations != -1, axis=1)
            N_withcrowd = np.sum(crowd_labels_present)

            no_crowd_labels = np.invert(crowd_labels_present)
            N_nocrowd = np.sum(no_crowd_labels)

            # check whether the additional test set needs to be split from the main dataset
            if N_nocrowd == 0:
                test_no_crowd = False
            else:
                test_no_crowd = True

                annotations = annotations[crowd_labels_present, :]
                ground_truth_nocrowd = ground_truth[no_crowd_labels]
                ground_truth = ground_truth[crowd_labels_present]

                text_nocrowd = text[no_crowd_labels]
                text = text[crowd_labels_present]

                doc_start_nocrowd = doc_start[no_crowd_labels]
                doc_start = doc_start[crowd_labels_present]


        # for the additional test set with no crowd labels
        scores_nocrowd = np.zeros((len(self.SCORE_NAMES), len(self.methods)))
        score_std_nocrowd = np.zeros((len(self.SCORE_NAMES)-3, len(self.methods)))

        preds_allmethods_nocrowd = -np.ones((annotations.shape[0], len(self.methods)))
        probs_allmethods_nocrowd = -np.ones((annotations.shape[0], self.num_classes, len(self.methods)))

        # timestamp for when we started a run. Can be compared to file versions to check what was run.
        timestamp = 'started-%Y-%m-%d-%H-%M-%S'.format(datetime.datetime.now())

        model = None # pass out to other functions if required

        for method_idx in range(len(self.methods)): 
            
            print('running method: %s' % self.methods[method_idx])

            method = self.methods[method_idx]

            # Initialise the results because not all methods will fill these in
            if test_no_crowd:
                agg_nocrowd = np.zeros(N_nocrowd)
                probs_nocrowd = np.ones((N_nocrowd, self.num_classes))

            if method == 'best':
                agg, probs = self._run_best_worker(annotations, ground_truth, doc_start)

            elif method == 'worst':
                agg, probs = self._run_worst_worker(annotations, ground_truth, doc_start)

            elif method == 'majority':
                agg, probs = majority_voting.MajorityVoting(annotations, self.num_classes).vote()
            
            elif method == 'clustering':
                agg, probs = self._run_clustering(annotations, doc_start, ground_truth)
                
            elif method == 'mace':
                agg, probs = self._run_mace(anno_path, ground_truth)
                
            elif method == 'ibcc':
                agg, probs = self._run_ibcc(annotations)
                
            elif method.split('_')[0] == 'bac':
                agg, probs, model = self._run_bac(annotations, doc_start, text, method)
                
            elif 'HMM_crowd' in method:
                sentences, crowd_labels, nfeats = self.data_to_hmm_crowd_format(annotations, text, doc_start, outputdir)
                
                data = crowd_data(sentences, crowd_labels)
                hc = HMM_crowd(self.num_classes, nfeats, data, None, None, vb=[0.1, 0.1], ne=1)
                hc.init(init_type = 'dw', wm_rep='cv2', dw_em=5, wm_smooth=0.1)
                hc.em(20)
                hc.mls()
                #agg = np.array(hc.res).flatten()
                agg = np.concatenate(hc.res)[:,None]
                probs = []
                for sentence_post_arr in hc.sen_posterior:
                    for tok_post_arr in sentence_post_arr:
                        probs.append(tok_post_arr)
                probs = np.array(probs)

            if '_then_LSTM' in method:
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
                                                                               agg.flatten())

                if ground_truth_val is None or doc_start_val is None or text_val is None:
                    # If validation set is unavailable, select a random subset of combined data to use for validation
                    # Simulates a scenario where all we have available are crowd labels.
                    train_sentences, dev_sentences = lstm_wrapper.split_train_to_dev(labelled_sentences)
                else:
                    train_sentences = labelled_sentences
                    dev_sentences, _, _ = lstm_wrapper.data_to_lstm_format(len(ground_truth_val), text_val,
                                                                        doc_start_val, ground_truth_val)

                lstm, f_eval, _ = lstm_wrapper.train_LSTM(train_sentences, dev_sentences)

                # now make predictions for all sentences
                agg, probs = lstm_wrapper.predict_LSTM(lstm, labelled_sentences, f_eval, IOB_map, self.num_classes)

                if test_no_crowd:
                    labelled_sentences, IOB_map, _ = lstm_wrapper.data_to_lstm_format(N_nocrowd, text_nocrowd,
                                                                           doc_start_nocrowd, np.zeros(N_nocrowd)-1)

                    agg_nocrowd, probs_nocrowd = lstm_wrapper.predict_LSTM(lstm, labelled_sentences, f_eval, IOB_map,
                                                                         self.num_classes)


            if np.any(ground_truth != -1): # don't run this in the case that crowd-labelled data has no gold labels
                scores_allmethods[:,method_idx][:,None], \
                score_std_allmethods[:, method_idx] = self.calculate_scores(agg, ground_truth.flatten(), probs,
                                                                                doc_start)
                preds_allmethods[:, method_idx] = agg.flatten()
                probs_allmethods[:,:,method_idx] = probs

            if test_no_crowd:
                scores_nocrowd[:,method_idx][:,None], \
                score_std_nocrowd[:, method_idx] = self.calculate_scores(agg_nocrowd,
                                                    ground_truth_nocrowd.flatten(), probs_nocrowd, doc_start_nocrowd)
                preds_allmethods_nocrowd[:, method_idx] = agg_nocrowd.flatten()
                probs_allmethods_nocrowd[:,:,method_idx] = probs_nocrowd

            print('...done')

            # Save the results so far after each method has completed.
            if outputdir is not None:
                if not os.path.exists(outputdir):
                    os.mkdir(outputdir)

                np.savetxt(outputdir + 'result_%s.csv' % timestamp, scores_allmethods, fmt='%s', delimiter=',',
                           header=str(self.methods).strip('[]'))
                np.savetxt(outputdir + 'pred_%s.csv' % timestamp, preds_allmethods, fmt='%s', delimiter=',',
                           header=str(self.methods).strip('[]'))
                with open(outputdir + 'probs_%s.pkl' % timestamp, 'wb') as fh:
                    pickle.dump(probs_allmethods, fh)

                np.savetxt(outputdir + 'result_nocrowd_%s.csv' % timestamp, scores_nocrowd, fmt='%s', delimiter=',',
                           header=str(self.methods).strip('[]'))
                np.savetxt(outputdir + 'pred_nocrowd_%s.csv' % timestamp, preds_allmethods_nocrowd, fmt='%s', delimiter=',',
                           header=str(self.methods).strip('[]'))
                with open(outputdir + 'probs_nocrowd_%s.pkl' % timestamp, 'wb') as fh:
                    pickle.dump(probs_allmethods_nocrowd, fh)

        if test_no_crowd:
            if return_model:
                return scores_allmethods, preds_allmethods, probs_allmethods, model, scores_nocrowd, \
                       preds_allmethods_nocrowd, probs_allmethods_nocrowd
            else:
                return scores_allmethods, preds_allmethods, probs_allmethods, scores_nocrowd, \
                       preds_allmethods_nocrowd, probs_allmethods_nocrowd
        else:
            if return_model:
                return scores_allmethods, preds_allmethods, probs_allmethods, model
            else:
                return scores_allmethods, preds_allmethods, probs_allmethods

    def data_to_hmm_crowd_format(self, annotations, text, doc_start, outputdir):

        filename = outputdir + '/hmm_crowd_text_data.pkl'

        annotations = annotations[:10000]
        text = text[:10000]
        doc_start = doc_start[:10000]

        if not os.path.exists(filename):
            if text is not None:

                ufeats = []
                features = np.zeros(len(text))
                for t, tok in enumerate(text):

                    print('converting data to hmm format: token %i / %i' % (t, len(text)))

                    if tok not in ufeats:
                        features[t] = len(ufeats)

                        ufeats.append(tok)
                    else:
                        features[t] = np.argwhere(ufeats == tok)[0][0]
            else:
                features = []
                ufeats = []

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
                        worker_labs = crowdlab(worker, len(sentences_inst) - 1, [int(annotations[i, worker])])

                        crowd_labs.append(worker_labs)
                    crowd_labels.append(crowd_labs)

                else:
                    sentence_inst.append(instance(token_feature_vector, label + 1))

                    for worker in range(annotations.shape[1]):
                        crowd_labs[worker].sen.append(int(annotations[i, worker]))

            nfeats = len(ufeats)

            with open(filename, 'wb') as fh:
                pickle.dump([sentences_inst, crowd_labels, nfeats], fh)
        else:
            with open(filename, 'rb') as fh:
                data = pickle.load(fh)
            sentences_inst = data[0]
            crowd_labels = data[1]
            nfeats = data[2]

        return sentences_inst, crowd_labels, nfeats

    def calculate_sample_metrics(self, agg, gt, probs):

        result = -np.ones(6)

        result[0] = skm.accuracy_score(gt, agg)
        result[1] = skm.precision_score(gt, agg, average='macro')
        result[2] = skm.recall_score(gt, agg, average='macro')
        result[3] = skm.f1_score(gt, agg, average='macro')

        auc_score = 0
        for i in range(probs.shape[1]):
            auc_i = skm.roc_auc_score(gt == i, probs[:, i])
            # print 'AUC for class %i: %f' % (i, auc_i)
            auc_score += auc_i * np.sum(gt == i)
        result[4] = auc_score / float(gt.shape[0])
        result[5] = skm.log_loss(gt, probs, eps=1e-100)

        return result

    def calculate_scores(self, agg, gt, probs, doc_start):
        
        result = -np.ones((9,1))        
        result[7] = metrics.num_invalid_labels(agg, doc_start)        

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
        print(conf)

        N = len(agg)
        if N < 50000:
            # use bootstrap resampling with small datasets
            nbootstraps = 100
            nmetrics = 6
            resample_results = np.zeros((nmetrics, nbootstraps))

            for i in range(nbootstraps):

                sampleidxs = np.random.choice(N, N, replace=True)

                resample_results[:, i] = self.calculate_sample_metrics(agg[sampleidxs],
                                                                       gt[sampleidxs],
                                                                       probs[sampleidxs])

            result[:6, 0] = np.mean(resample_results, axis=1)
            std_result = np.std(resample_results, axis=1)

        else:
            result[:6, 0] = self.calculate_sample_metrics(agg, gt, probs)
            result[6] = metrics.abs_count_error(agg, gt)
            result[8] = metrics.mean_length_error(agg, gt, doc_start)

            std_result = None

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
                results[param_idx,:,:,run_idx], preds, probabilities = self.run_methods(annos, gt, doc_start, param_idx, data_path + 'annotations.csv')
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
