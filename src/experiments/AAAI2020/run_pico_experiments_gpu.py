'''
Created on April 27, 2018

@author: Edwin Simpson
'''
import os

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = os.path.join(load_data.output_root_dir, 'pico3')
regen_data = False

gt, annos, doc_start, features, gt_val, _, _, _ = load_data.load_biomedical_data(regen_data) #, debug_subset_size=1000)

beta0_factor = 1
alpha0_diags = 10
alpha0_factor = 10
best_begin_factor = 10
exp = Experiment(output_dir, 3, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
                 alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor,
                 max_iter=20, begin_factor=best_begin_factor)
# # run all the methods that don't require tuning here
exp.methods = [
                # 'bac_seq_integrateIF',
                'bac_seq_integrateIF_thenLSTM',
                'bac_seq_integrateIF_integrateLSTM_atEnd',
]
# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(new_data=regen_data)

# ------------------------------------------------------------------------------------------------
beta0_factor = 100
alpha0_diags = 1
alpha0_factor = 1
exp = Experiment(output_dir, 3, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
                 alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor,
                 max_iter=20)
exp.methods =  [
                'gt_thenLSTM',
                'HMM_crowd_thenLSTM',
]
# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(new_data=regen_data)