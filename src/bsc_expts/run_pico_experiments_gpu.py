'''
Created on April 27, 2018

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_sequence_combination/output/pico/'

regen_data = False

# print('USING ONLY A SUBSET OF DATA FOR DEBUGGING!!!!!!!!!!!')
gt, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev = \
    load_data.load_biomedical_data(regen_data)

exp = Experiment(None, 3, annos.shape[1], None, max_iter=20)

exp.save_results = True
exp.opt_hyper = False #True

exp.alpha0_diags = 100
exp.alpha0_factor = 9

best_bac_wm = 'bac_seq' # choose model with best score for the different BAC worker models
best_nu0factor = 100
best_diags = 1
best_factor = 1

nu_factors = [0.1, 10, 100]
diags = [0.1, 1, 10, 100]
factors = [0.1, 1, 10]

#  ------------------------------------------------------------------------------------------------
# exp = Experiment(None, 3, annos.shape[1], None, max_iter=20)
#
# exp.save_results = True
# exp.opt_hyper = False
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
#
# exp.methods =  [
#                 best_bac_wm + '_integrateIF_then_LSTM',
#                 #'HMM_crowd_then_LSTM',
# ]
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                 new_data=regen_data
#                 )

# reset ------------------------------------------------------------------------------------------------
exp = Experiment(None, 3, annos.shape[1], None, max_iter=20)

exp.save_results = True
exp.opt_hyper = False

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor

exp.methods =  [
                best_bac_wm + '_integrateIF_integrateLSTM_atEnd',
]

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text,
                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                new_data=regen_data
                )
