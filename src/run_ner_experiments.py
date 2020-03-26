'''
Created on April 27, 2018

@author: Edwin Simpson
'''
import os

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = os.path.join(load_data.output_root_dir, 'ner')

regen_data = False
gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val, _ = \
    load_data.load_ner_data(regen_data)


# # -------------------- debug with subset -------
# s = 100
# idxs = np.argwhere(gt!=-1)[:s, 0]
# gt = gt[idxs]
# annos = annos[idxs]
# doc_start = doc_start[idxs]
# text = text[idxs]
# gt_task1_val = gt_task1_val[idxs]
#
# # -------------------------
#
# exp = Experiment(None, 9, annos.shape[1], None, max_iter=20)
#
# nu_factors = [0.1, 10, 100]
# diags = [0.1, 1, 10, 100]
# factors = [0.1, 1, 10, 100]
# methods_to_tune = [
#                    'bac_ibcc_integrateIF_noHMM',
#                    'bac_seq_integrateIF_noHMM',
#                    'bac_ibcc',
#                    'bac_mace'
#                    'ibcc',
#                    'HMM_crowd',
#                    'bac_vec_integrateIF',
#                    'bac_ibcc_integrateIF',
#                    'bac_seq_integrateIF',
#                    'bac_acc_integrateIF',
#                    'bac_mace_integrateIF'
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
#     print('Best values: %f, %f, %f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
#
#     # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
#     exp.methods = [method]
#     exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, return_model=False,
#                 ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#                 new_data=regen_data
#                 )

# ------------------------------------------------------------------------------------------------
# Rerunning with found parameters...
#
#
# niter = 20 # variational inference iterations
#
# exp = Experiment(None, 9, annos.shape[1], None, max_iter=niter)
# exp.save_results = True
# exp.opt_hyper = False#True
#
# exp.nu0_factor = 1
# exp.alpha0_diags = 10
# exp.alpha0_factor = 10
# output_dir = os.path.join(load_data.output_root_dir, 'ner3_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
#
# exp.methods =  [
#                 'bac_vec_integrateIF',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(
#     annos, gt, doc_start, output_dir, text,
#     ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#     new_data=regen_data
# )
#
# #----------------------------------------------------------------------------
# niter = 20 # variational inference iterations
#
# exp = Experiment(None, 9, annos.shape[1], None, max_iter=niter)
# exp.save_results = True
# exp.opt_hyper = False#True
#
# exp.nu0_factor = 1
# exp.alpha0_diags = 100
# exp.alpha0_factor = 10
# output_dir = os.path.join(load_data.output_root_dir, 'ner3_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
#
# exp.methods =  [
#                 'bac_ibcc_integrateIF',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(
#     annos, gt, doc_start, output_dir, text,
#     ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#     # ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
#     ground_truth_nocrowd=None, doc_start_nocrowd=None, text_nocrowd=None,
#     new_data=regen_data
# )
#
# #----------------------------------------------------------------------------
niter = 20 # variational inference iterations

exp = Experiment(None, 9, annos.shape[1], None, max_iter=niter)
exp.save_results = True
exp.opt_hyper = False#True

exp.nu0_factor = 1
exp.alpha0_diags = 1#100
exp.alpha0_factor = 10
output_dir = os.path.join(load_data.output_root_dir, 'ner3_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))

exp.methods =  [
                'majority',
                'worst',
                'best',
                'bac_seq_integrateIF',
]

# should run both task 1 and 2.
exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    # ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    new_data=regen_data
)
#
