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

regen_data = False

datadir = 'bio'
gt, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev = \
    load_data.load_biomedical_data(regen_data, data_folder=datadir) # , debug_subset_size=50000)

best_nu0factor = 1
best_diags = 10
best_factor = 10
best_begin_factor = 10

exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, begin_factor=best_begin_factor)

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor

# # run all the methods that don't require tuning here
exp.methods =  [
                # 'bac_seq_integrateIF',
                'bac_seq_integrateIF_then_LSTM',
                'bac_seq_integrateIF_integrateLSTM_atEnd',
]

output_dir = os.path.join(load_data.output_root_dir, 'pico3')
#_wormulon_%f_%f_%f_%f_%s_betaOO_nu0_1_prior' (best_nu0factor, best_diags, best_factor, best_begin_factor, datadir))

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text,
                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                 new_data=regen_data
                 )


# # ------------------------------------------------------------------------------------------------
# exp = Experiment(None, 3, annos.shape[1], None, max_iter=20)
#
# exp.save_results = True
# exp.opt_hyper = False
#
# best_nu0factor = 100
# best_diags = 1
# best_factor = 1
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
#
# exp.methods =  [
#                 'gt_then_LSTM',
#                 'HMM_crowd_then_LSTM',
# ]
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                 new_data=regen_data
#                 )