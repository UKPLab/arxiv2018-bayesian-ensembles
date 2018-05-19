'''
Created on April 27, 2018

@author: Edwin Simpson
'''
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_annotator_combination/output/ner_al/'

gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val = \
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

exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1)
exp.save_results = True
exp.opt_hyper = False#True

# best IBCC setting so far is diag=100, factor=36. Let's reuse this for BIO and all BAC_IBCC runs.

best_bac_wm = 'bac_ibcc' #'unknown' # choose model with best score for the different BAC worker models

exp.alpha0_diags = 50 # best_diags
exp.alpha0_factor = 1#9 # best_factor

# run all the methods that don't require tuning here
exp.methods =  [
    'HMM_crowd' + '_then_LSTM',
    best_bac_wm + '_integrateLSTM',
                ]

results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(
                    annos, gt, doc_start, output_dir, text,
                    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
                    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
                    active_learning=True
)
