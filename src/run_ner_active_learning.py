'''
Created on April 27, 2018

@author: Edwin Simpson
'''
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np
import os

gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val, _ = \
    load_data.load_ner_data(False)

# debug with subset -------
# s = 100
# idxs = np.argwhere(gt!=-1)[:s, 0]
# gt = gt[idxs]
# annos = annos[idxs]
# doc_start = doc_start[idxs]
# text = text[idxs]
# gt_task1_val = gt_task1_val[idxs]
# -------------------------

output_dir = '../../data/bayesian_annotator_combination/output/ner_al_new/'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

exp = Experiment(None, 9, annos.shape[1], None, max_iter=10, crf_probs=True)
exp.save_results = True
exp.opt_hyper = False#True

exp.nu0_factor = 0.1
exp.alpha0_diags = 1 # best_diags
exp.alpha0_factor = 1#9 # best_factor

# run all the methods that don't require tuning here
exp.methods =  [
    'bac_seq_integrateBOF_integrateLSTM_atEnd',
    #'bac_seq_integrateBOF_then_LSTM',
    'HMM_crowd_then_LSTM'
                ]

results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(
                    annos, gt, doc_start, output_dir, text,
                    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
                    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
                    active_learning=True
)

# Random Sampling ------------------------------------------------------------------------------

output_dir = '../../data/bayesian_annotator_combination/output/ner_rand_new/'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

exp = Experiment(None, 9, annos.shape[1], None, max_iter=10, crf_probs=False)
exp.save_results = True
exp.opt_hyper = False#True

exp.nu0_factor = 0.1
exp.alpha0_diags = 1 # best_diags
exp.alpha0_factor = 1#9 # best_factor

# run all the methods that don't require tuning here
exp.methods =  [
    'bac_seq_integrateBOF_integrateLSTM_atEnd',
    'bac_seq_integrateBOF_then_LSTM',
    'HMM_crowd_then_LSTM'
                ]

exp.random_sampling = True

results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(
                    annos, gt, doc_start, output_dir, text,
                    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
                    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
                    active_learning=True
)
