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
# s = 100
# idxs = np.argwhere(gt_task1_val!=-1).flatten()[:s]
# gt = gt[idxs]
# annos = annos[idxs]
# doc_start = doc_start[idxs]
# text = text[idxs]
# gt_task1_val = gt_task1_val[idxs]
# -------------------------

exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1)
exp.save_results = True
exp.opt_hyper = False#True

diags = [1, 5, 10, 50]
factors = [1, 4, 9, 16, 25]
methods_to_tune = ['bac_mace', 'ibcc', 'bac_acc', 'bac_ibcc', 'bac_seq']

best_bac_wm = 'unknown' # choose model with best score for the different BAC worker models
best_bac_wm_score = -np.inf

for m, method in enumerate(methods_to_tune):
    print('TUNING %s' % method)

    best_scores = exp.tune_alpha0(diags, factors, [method], annos, gt_task1_val, doc_start, output_dir, text)
    best_idxs = np.unravel_index(np.argmax(best_scores), best_scores.shape)
    exp.alpha0_diags = diags[best_idxs[0]]
    exp.alpha0_factor = factors[best_idxs[1]]

    print('Best values: %f, %f' % (exp.alpha0_diags, exp.alpha0_factor))

    # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
    exp.methods = [method]
    exp.run_methods(annos, gt, doc_start, output_dir, text,
                ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val)

    best_score = np.max(best_scores)
    if 'bac' in method and best_score > best_bac_wm_score:
        best_bac_wm = method
        best_bac_wm_score = best_score
        best_diags = exp.alpha0_diags
        best_factor = exp.alpha0_factor

print('best BAC method = %s' % best_bac_wm)

# run all the methods that don't require tuning here
exp.methods =  ['majority', 'best', 'worst', 'HMM_crowd', 'HMM_crowd_then_LSTM',
                best_bac_wm + '_then_LSTM', best_bac_wm + '_integrateLSTM']

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor

results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(annos, gt, doc_start, output_dir,
                                       text, ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val)
# for now we are just running task 1. Then we choose which version of BAC to run for task 2 with LSTM, change the
# methods and uncomment the code below.
#, gt_nocrowd, doc_start_nocrowd, text_nocrowd)
