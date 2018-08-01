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
gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val, _ = \
    load_data.load_ner_data(regen_data)
#
# debug with subset -------
# s = 1000
# idxs = np.argwhere(gt!=-1)[:s, 0]
# gt = gt[idxs]
# annos = annos[idxs]
# doc_start = doc_start[idxs]
# print('No. documents:')
# print(np.sum(doc_start))
# text = text[idxs]
# gt_task1_val = gt_task1_val[idxs]
# -------------------------

exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1)
exp.save_results = True
exp.opt_hyper = False#True

# run all the methods that don't require tuning here
exp.methods =  [
                'HMM_crowd',
                'HMM_crowd_then_LSTM',
]

# should run both task 1 and 2.

exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    new_data=regen_data
)