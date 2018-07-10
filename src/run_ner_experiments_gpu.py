'''

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_annotator_combination/output/ner-by-sentence/'

regen_data = False
gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val = \
    load_data.load_ner_data(regen_data)

exp = Experiment(None, 9, annos.shape[1], None, max_iter=20)
exp.save_results = True
exp.opt_hyper = False#True

best_bac_wm = 'bac_seq' #'unknown' # choose model with best score for the different BAC worker models
best_bac_wm_score = -np.inf

best_nu0factor = 0.1
best_diags = 1
best_factor = 1
best_acc_bias = 0

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor
exp.alpha0_acc_bias = best_acc_bias

# run all the methods that don't require tuning here
exp.methods =  [
                best_bac_wm + '_integrateBOF_then_LSTM',
]

# should run both task 1 and 2.
exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    new_data=regen_data
)

# reset to free memory? ------------------------------------------------------------------------------------------------
exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1, max_iter=20)
exp.save_results = True
exp.opt_hyper = False#True

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor
exp.alpha0_acc_bias = best_acc_bias

# run all the methods that don't require tuning here
exp.methods =  [
                best_bac_wm + '_integrateBOF_integrateLSTM_atEnd',
]

# should run both task 1 and 2.

exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    new_data=regen_data
)

# reset to free memory? ------------------------------------------------------------------------------------------------
exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1, max_iter=20)
exp.save_results = True
exp.opt_hyper = False#True

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor
exp.alpha0_acc_bias = best_acc_bias

# run all the methods that don't require tuning here
exp.methods =  [
                'HMM_crowd_then_LSTM',
]

# should run both task 1 and 2.
exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    new_data=regen_data
)

# # reset to free memory? ------------------------------------------------------------------------------------------------
# exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1, max_iter=20)
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
#                 best_bac_wm + '_integrateBOF_integrateLSTM',
# ]
#
# # should run both task 1 and 2.
#
# exp.run_methods(
#     annos, gt, doc_start, output_dir, text,
#     ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#     ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
#     new_data=regen_data
# )
#
# # reset to free memory? ------------------------------------------------------------------------------------------------
# exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1, max_iter=20)
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
#                 best_bac_wm + '_integrateLSTM',
# ]
#
# # should run both task 1 and 2.
#
# exp.run_methods(
#     annos, gt, doc_start, output_dir, text,
#     ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#     ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
#     new_data=regen_data
# )
#
#
# # reset to free memory? ------------------------------------------------------------------------------------------------
# exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1, max_iter=20)
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
#                 best_bac_wm + '_integrateLSTM_integrateBOF_atEnd_noHMM',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(
#     annos, gt, doc_start, output_dir, text,
#     ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#     ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
#     new_data=regen_data
# )