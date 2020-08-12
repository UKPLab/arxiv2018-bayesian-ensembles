'''
Created on April 27, 2018

@author: Edwin Simpson
'''
import os

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np


regen_data = False
gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_val, _ = \
    load_data.load_ner_data(regen_data)

# -------------------- debug or tune with subset -------
s = 500
idxs = np.argwhere(gt != -1)[:, 0] # for testing
ndocs = np.sum(doc_start[idxs])

if ndocs > s:
    idxs = idxs[:np.argwhere(np.cumsum(doc_start[idxs])==s)[0][0]]
elif ndocs < s:  # not enough validation data
    moreidxs = np.argwhere(gt != -1)[:, 0]
    deficit = s - ndocs
    ndocs = np.sum(doc_start[moreidxs])
    if ndocs > deficit:
        moreidxs = moreidxs[:np.argwhere(np.cumsum(doc_start[moreidxs])==deficit)[0][0]]
    idxs = np.concatenate((idxs, moreidxs))

gt = gt[idxs]
annos = annos[idxs]
doc_start = doc_start[idxs]
text = text[idxs]
gt_val = gt_val[idxs]

# ------------------------- HYPERPARAMETER TUNING ----------------------

# # tune with small dataset to save time
# s = 250
# idxs = np.argwhere(gt_val != -1)[:, 0] # for tuning
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
# gt = gt[idxs]
# annos_val = annos[idxs]
# doc_start_val = doc_start[idxs]
# text_val = text[idxs]
# gt_val = gt_val[idxs]
#
# output_dir = os.path.join(load_data.output_root_dir, 'ner')
# exp = Experiment(output_dir, 9, annos, gt, doc_start, text, annos_val, gt_val, doc_start_val, text_val, max_iter=20)
# beta_factors = [0.1, 10, 100]
# diags = [0.1, 1, 10, 100]
# factors = [0.1, 1, 10, 100]
# methods_to_tune = [
#                    'bac_ibcc_integrateIF_noHMM',
#                    'bac_seq_integrateIF_noHMM',
#                    'bac_ibcc',
#                    'bac_seq',
#                    'ibcc',
#                    'HMM_crowd',
#                    'bac_vec_integrateIF',
#                    'bac_ibcc_integrateIF',
#                    'bac_seq_integrateIF',
#                    'bac_acc_integrateIF',
#                    'bac_mace_integrateIF'
#                    ]
# #
# for m, method in enumerate(methods_to_tune):
#     print('TUNING %s' % method)
#
#     best_scores = exp.tune_alpha0(diags, factors, beta_factors, method)
#
#     best_idxs = best_scores[1:].astype(int)
#     exp.beta0_factor = beta_factors[best_idxs[0]]
#     exp.alpha0_diags = diags[best_idxs[1]]
#     exp.alpha0_factor = factors[best_idxs[2]]
#
#     print('Best values: %f, %f, %f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
#
#     # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
#     exp.methods = [method]
#     exp.run_methods(new_data=regen_data)

# ------------------------------------------------------------------------------------------------
# Rerunning with found parameters...
# beta0_factor = 1
# alpha0_diags = 10
# alpha0_factor = 10
# output_dir = os.path.join(load_data.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags, alpha0_factor))
# exp = Experiment(output_dir, 9, annos, gt, doc_start, text, annos, gt_val, doc_start, text,
#                   alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
#
# exp.methods =  [
#                 'bac_vec_integrateIF',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(new_data=regen_data)
#
# #----------------------------------------------------------------------------
# beta0_factor = 1
# alpha0_diags = 100
# alpha0_factor = 10
# output_dir = os.path.join(load_data.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags, alpha0_factor))
# exp = Experiment(output_dir, 9, annos, gt, doc_start, text, annos, gt_val, doc_start, text,
#                   alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
#
# exp.methods =  [
#                 'bac_ibcc_integrateIF',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(new_data=regen_data)
#
# #----------------------------------------------------------------------------
#
# beta0_factor = 0.1
# alpha0_diags = 0.1#100
# alpha0_factor = 1
# output_dir = os.path.join(load_data.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags, alpha0_factor))
# exp = Experiment(output_dir, 9, annos, gt, doc_start, text, annos, gt_val, doc_start, text,
#                   alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
#
# exp.methods =  [
#                 'ibcc',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(new_data=regen_data)

#----------------------------------------------------------------------------
beta0_factor = 1
alpha0_diags = 1#100
alpha0_factor = 10
output_dir = os.path.join(load_data.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags, alpha0_factor))
exp = Experiment(output_dir, 9, annos, gt, doc_start, text, annos, gt_val, doc_start, text,
                 gt_nocrowd, doc_start_nocrowd, text_nocrowd,
                 alpha0_factor, alpha0_diags, beta0_factor,
                 max_iter=20)
exp.opt_hyper = False#True

exp.methods =  [
                'majority',
                'worst',
                'best',
                'bac_seq_integrateIF',
]

# should run both task 1 and 2.
exp.run_methods(new_data=regen_data)

