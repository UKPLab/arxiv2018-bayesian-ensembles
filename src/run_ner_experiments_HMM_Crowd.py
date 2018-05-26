'''
Created on April 27, 2018

@author: Edwin Simpson
'''
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np
# import pandas as pd
# from data.load_data import _map_ner_str_to_labels

# TODO: why do the BAC methods fail to work properly?
# TODO: rerun the BAC methods.

output_dir = '../../data/bayesian_annotator_combination/output/ner-by-sentence/'

regen_data = False
gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val = \
    load_data.load_ner_data(regen_data)

exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1)
exp.save_results = True
exp.opt_hyper = False#True

# best IBCC setting so far is diag=100, factor=36. Let's reuse this for BIO and all BAC_IBCC runs.

exp.alpha0_diags = 50
exp.alpha0_factor = 1

# run all the methods that don't require tuning here
exp.methods =  [
                'HMM_crowd',
                'HMM_crowd_then_LSTM',
]

# should run both task 1 and 2.

results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    new_data=regen_data
)