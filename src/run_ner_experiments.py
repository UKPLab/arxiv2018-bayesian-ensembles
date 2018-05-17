'''
Created on April 27, 2018

@author: Edwin Simpson
'''
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_annotator_combination/output/ner/'

gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val = \
    load_data.load_ner_data(False)

# debug with subset -------
s = 100
idxs = np.argwhere(gt!=-1)[:s, 0]
gt = gt[idxs]
annos = annos[idxs]
doc_start = doc_start[idxs]
text = text[idxs]
gt_task1_val = gt_task1_val[idxs]
# -------------------------

exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1)
exp.save_results = True
exp.opt_hyper = False#True

# best IBCC setting so far is diag=100, factor=36. Let's reuse this for BIO and all BAC_IBCC runs.

diags = [1, 50, 100]#[1, 50, 100]#[1, 5, 10, 50]
factors = [1, 4, 9, 36]#[36, 49, 64]#[1, 4, 9, 16, 25]
methods_to_tune = ['bac_mace', 'bac_seq', 'bac_ibcc', 'bac_acc', 'ibcc']

best_bac_wm = 'bac_ibcc' #'unknown' # choose model with best score for the different BAC worker models
best_bac_wm_score = -np.inf

# # tune with small dataset to save time
# s = 250
# idxs = np.argwhere(gt_task1_val != -1)[:, 0]
# ndocs = np.sum(doc_start[idxs])
#
# if ndocs > s:
#     idxs = idxs[:np.argwhere(np.cumsum(doc_start[idxs])==s)[0][0]]
# elif ndocs < s:  # not enough validation data
#     moreidxs = np.argwhere(gt != -1)[:, 0]
#     deficit = s - ndocs
#     ndocs = np.sum(doc_start[moreidxs])
#     if ndocs > deficit:
#         moreidxs = moreidxs[:np.argwhere(np.cumsum(doc_start[moreidxs])==deficit)[0][0]]
#     idxs = np.concatenate((idxs, moreidxs))
#
# tune_gt = gt[idxs]
# tune_annos = annos[idxs]
# tune_doc_start = doc_start[idxs]
# tune_text = text[idxs]
# tune_gt_task1_val = gt_task1_val[idxs]
#
# for m, method in enumerate(methods_to_tune):
#     print('TUNING %s' % method)
#
#     best_scores = exp.tune_alpha0(diags, factors, method, tune_annos, tune_gt_task1_val, tune_doc_start,
#                                   output_dir, tune_text)
#     best_idxs = best_scores[1:].astype(int)
#     exp.alpha0_diags = diags[best_idxs[0]]
#     exp.alpha0_factor = factors[best_idxs[1]]
#
#     print('Best values: %f, %f' % (exp.alpha0_diags, exp.alpha0_factor))
#
#     # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
#     exp.methods = [method]
#     exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True,
#                 ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val)
#
#     best_score = np.max(best_scores)
#     if 'bac' in method and best_score > best_bac_wm_score:
#         best_bac_wm = method
#         best_bac_wm_score = best_score
#         best_diags = exp.alpha0_diags
#         best_factor = exp.alpha0_factor
#
# print('best BAC method tested here = %s' % best_bac_wm)
#
# # stuff to rerun
# exp.alpha0_diags = 1 # best_diags
# exp.alpha0_factor = 1 # best_factor
#
# # run all the methods that don't require tuning here
# exp.methods =  ['bac_seq']
#
# results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(annos, gt, doc_start, output_dir,
#                                        text, ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val)

best_bac_wm = 'bac_ibcc'

exp.alpha0_diags = 50 # best_diags
exp.alpha0_factor = 1#9 # best_factor

# run all the methods that don't require tuning here
exp.methods =  [#'majority', 'best', 'worst',
                'HMM_crowd'#, 'HMM_crowd_then_LSTM',
                #best_bac_wm + '_then_LSTM', best_bac_wm + '_integrateLSTM'
                #'bac_mace', 'bac_acc', , 'bac_seq', 'ibcc'
                ]

results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(annos, gt, doc_start, output_dir,
                                       text, ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val)

# for now we are just running task 1. Then we choose which version of BAC to run for task 2 with LSTM, change the
# methods and uncomment the code below.
#, gt_nocrowd, doc_start_nocrowd, text_nocrowd)
