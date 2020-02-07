'''

@author: Edwin Simpson
'''
import os

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = os.path.join(load_data.output_root_dir, 'ner3')

regen_data = False
gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val, gt_all = \
    load_data.load_ner_data(regen_data)

#-------------------------------------------------------------------------------------
niter = 20 # variational inference iterations

exp = Experiment(None, 9, annos.shape[1], None, max_iter=niter)
exp.save_results = True
exp.opt_hyper = False#True
# exp.use_lb = True

exp.nu0_factor = 1
exp.alpha0_diags = 1
exp.alpha0_factor = 10
output_dir = os.path.join(load_data.output_root_dir, 'ner3')  #_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
exp.methods =  [
                # 'bac_seq_integrateIF',
                'bac_seq_integrateIF_then_LSTM',
                'bac_seq_integrateIF_integrateLSTM_atEnd',
]

# should run both task 1 and 2.
exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    # ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    ground_truth_nocrowd=None, doc_start_nocrowd=None, text_nocrowd=None,
    new_data=regen_data
)

#-------------------------------------------------------------------------------------
niter = 20 # variational inference iterations

exp = Experiment(None, 9, annos.shape[1], None, max_iter=niter)
exp.save_results = True
exp.opt_hyper = False#True
# exp.use_lb = True

exp.nu0_factor = 1
exp.alpha0_diags = 10
exp.alpha0_factor = 10
output_dir = os.path.join(load_data.output_root_dir, 'ner3')  #_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
exp.methods =  [
                # 'bac_seq_integrateIF',
                'bac_seq_integrateIF_then_LSTM',
                'bac_seq_integrateIF_integrateLSTM_atEnd',
]

# should run both task 1 and 2.
exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    # ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    ground_truth_nocrowd=None, doc_start_nocrowd=None, text_nocrowd=None,
    new_data=regen_data
)

# exp = Experiment(None, 9, annos.shape[1], None, max_iter=niter)
# exp.save_results = True
# exp.opt_hyper = False#True
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
# exp.alpha0_acc_bias = best_acc_bias
#
# # run all the methods that don't require tuning here
# exp.methods =  [
#                 'HMM_crowd_then_LSTM',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(
#     annos, gt, doc_start, output_dir, text,
#     ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#     ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
#     new_data=regen_data
# )

# # ------------------------------------------------------------------------------------------------
# exp = Experiment(None, 9, annos.shape[1], None, max_iter=niter)
# exp.save_results = True
# exp.opt_hyper = False#True
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
# exp.alpha0_acc_bias = best_acc_bias
#
# #run all the methods that don't require tuning here
# exp.methods =  [
#                 'gt_then_LSTM', # train the LSTM on the real gold labels
# ]
#