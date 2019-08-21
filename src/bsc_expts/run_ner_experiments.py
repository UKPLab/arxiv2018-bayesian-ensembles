'''
Created on April 27, 2018

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_sequence_combination/output/ner/'

regen_data = False
gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val, _ = \
    load_data.load_ner_data(regen_data)


# ------------------------------------------------------------------------------------------------
best_nu0factor = 0.1 # changed for MACE...
best_diags = 0.1
best_factor = 0.1
#
# exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1, max_iter=20)
# exp.save_results = True
# exp.opt_hyper = False#True
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
#
# # run all the methods that don't require tuning here
# exp.methods =  [
#     'HMM_crowd',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(
#     annos, gt, doc_start, output_dir, text,
#     ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#     ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
#     new_data=regen_data
# )


exp = Experiment(None, 9, annos.shape[1], None, max_iter=20)
exp.save_results = True
exp.opt_hyper = False#True

best_bac_wm = 'bac_seq' #'unknown' # choose model with best score for the different BAC worker models
best_bac_wm_score = -np.inf

best_nu0factor = 0.1 # changed for MACE...
best_diags = 1
best_factor = 1
best_acc_bias = 0

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor
exp.alpha0_acc_bias = best_acc_bias

# run all the methods that don't require tuning here
exp.methods =  [
                'majority',
                # 'mace',
                # 'ds',
                # 'best',
                # 'worst',
                # best_bac_wm,
                # 'bac_mace_integrateIF'
]

# should run both task 1 and 2.
exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    new_data=regen_data
)
#
#
# # exp.alpha0_diags_lstm = 0.1
# # exp.alpha0_factor_lstm = 0.1
# #
# nu_factors = [0.1, 10, 100]
# diags = [0.1, 1, 10, 100]
# factors = [0.1, 1, 10]
# # # acc_biases = [1, 10, 100]
# #
# methods_to_tune = [
#                     #'bac_ibcc_integrateIF_noHMM',
#                     # 'bac_seq_integrateIF_noHMM',
#                     # 'bac_ibcc',
#                     # 'bac_mace'
# #                    'ibcc',
#                    'bac_vec_integrateIF',
# #                    'bac_ibcc_integrateIF',
# #                    'bac_seq_integrateIF',
# #                    'bac_acc_integrateIF',
# #                    'bac_mace_integrateIF'
#                    ]
#
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
#     best_scores = exp.tune_alpha0(diags, factors, nu_factors, method, tune_annos, tune_gt_task1_val, tune_doc_start,
#                                   output_dir, tune_text)
#
#     best_idxs = best_scores[1:].astype(int)
#     exp.nu0_factor = nu_factors[best_idxs[0]]
#     exp.alpha0_diags = diags[best_idxs[1]]
#     exp.alpha0_factor = factors[best_idxs[2]]
#
#     # exp.alpha0_diags_lstm = diags[best_idxs[1]]
#     # exp.alpha0_factor_lstm = factors[best_idxs[2]]
#
#     print('Best values: %f, %f, %f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
#
#     # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
#     exp.methods = [method]
#     exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, return_model=False,
#                 ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#                 ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd,
#                 text_nocrowd=text_nocrowd,
#                 new_data=regen_data
#                 )
#
#     # best_score = best_scores[0]
#     # if 'bac_seq' in method and best_score > best_bac_wm_score:
#     #     best_bac_wm = 'bac_' + method.split('_')[1]
#     #     best_bac_wm_score = best_score
#     #     best_diags = exp.alpha0_diags
#     #     best_factor = exp.alpha0_factor
#     #     best_nu0factor = exp.nu0_factor

# print('best BAC method tested here = %s' % best_bac_wm)
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
# exp.alpha0_acc_bias = best_acc_bias
#


# # ------------------------------------------------------------------------------------------------
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
#                 #'HMM_crowd',
#                 'bac_seq',
#                 'bac_seq_integrateIF_noHMM'
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(
#     annos, gt, doc_start, output_dir, text,
#     ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#     ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
#     new_data=regen_data
# )

# ------------------------------------------------------------------------------------------------
best_nu0factor = 0.1 # changed for MACE...
best_diags = 100
best_factor = 0.1

exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1, max_iter=20)
exp.save_results = True
exp.opt_hyper = False#True

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor
exp.alpha0_acc_bias = best_acc_bias

# run all the methods that don't require tuning here
exp.methods =  [
    'bac_ibcc_integrateIF',
    # 'bac_ibcc_integrateIF_noHMM',
    # 'bac_ibcc',
]

# should run both task 1 and 2.
exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    new_data=regen_data
)