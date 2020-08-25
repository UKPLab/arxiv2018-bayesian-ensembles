'''
Created on April 27, 2018

@author: Edwin Simpson
'''
import cProfile
import os

import evaluation.experiment
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np


regen_data = False
gt, annos, doc_start, features, gt_nocrowd, doc_start_nocrowd, features_nocrowd, gt_val, _ = \
    load_data.load_ner_data(regen_data)

# -------------------- debug or tune with subset -------
# s = 50
# idxs = np.argwhere(gt != -1)[:, 0] # for testing
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
# annos = annos[idxs]
# doc_start = doc_start[idxs]
# features = features[idxs]
# gt_val = gt_val[idxs]

# # ------------------------------------------------------------------------------------------------
# # Rerunning with found parameters...
#
# beta0_factor = 0.1
# alpha0_diags = 0.1
# alpha0_factor = 0.1
# output_dir = os.path.join(evaluation.experiment.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags,
#                                                                                     alpha0_factor))
# exp = Experiment(output_dir, 9, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
#                  alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
#
# exp.methods = [
#                 'best',  # does not use the hyperparameters
#                 'worst',  # does not use the hyperparameters
#                 'majority',  # does not use the hyperparameters
#                 'mace',  # worked best with its own default hyperparameters, smoothing=0.001, alpha=0.5, beta=0.5
#                 'ds',   # does not use the hyperparameters
#                 'HMM_Crowd',  # does not use alpha0_diags
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(new_data=regen_data)
#
# # ----------------------------------------------------------------------------
#
# beta0_factor = 0.1
# alpha0_diags = 0.1
# alpha0_factor = 1
# output_dir = os.path.join(evaluation.experiment.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags,
#                                                                                     alpha0_factor))
# exp = Experiment(output_dir, 9, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
#                  alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
# exp.methods = [
#                 'ibcc',
# ]
# exp.opt_hyper = False
# exp.run_methods(new_data=regen_data)
#
# # ----------------------------------------------------------------------------
#
# beta0_factor = ?
# alpha0_diags = ?
# alpha0_factor = 10
# output_dir = os.path.join(evaluation.experiment.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags,
#                                                                                     alpha0_factor))
# exp = Experiment(output_dir, 9, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
#                   alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
#
# exp.methods = [
#                 'bsc_acc_integrateIF',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(new_data=regen_data)
#
# # ----------------------------------------------------------------------------
#
# beta0_factor = ?
# alpha0_diags = ?
# alpha0_factor = ?
# output_dir = os.path.join(evaluation.experiment.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags,
#                                                                                     alpha0_factor))
# exp = Experiment(output_dir, 9, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
#                   alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
#
# exp.methods = [
#                 'bsc_spam_integrateIF',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(new_data=regen_data)

# ----------------------------------------------------------------------------

beta0_factor = 1
alpha0_diags = 10
alpha0_factor = 10
output_dir = os.path.join(evaluation.experiment.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags,
                                                                                    alpha0_factor))
exp = Experiment(output_dir, 9, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
                  alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)

exp.methods = [
                'bsc_cv_integrateIF',
]

# should run both task 1 and 2.
exp.run_methods(new_data=regen_data)

# ----------------------------------------------------------------------------

beta0_factor = 1
alpha0_diags = 100
alpha0_factor = 10
output_dir = os.path.join(evaluation.experiment.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags,
                                                                                    alpha0_factor))
exp = Experiment(output_dir, 9, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
                  alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)

exp.methods = [
                'bsc_cm_integrateIF',
]

# should run both task 1 and 2.
exp.run_methods(new_data=regen_data)

# # ----------------------------------------------------------------------------
#
# beta0_factor = 1
# alpha0_diags = 10
# alpha0_factor = 10
# output_dir = os.path.join(evaluation.experiment.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags,
#                                                                                     alpha0_factor))
# exp = Experiment(output_dir, 9, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
#                  alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
#
# exp.methods = [
#                 'bsc_seq_integrateIF',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(new_data=regen_data)

# ------------------------- HYPERPARAMETER TUNING ----------------------

# tune with small dataset to save time
s = 250
idxs = np.argwhere(gt_val != -1)[:, 0]  # for tuning
ndocs = np.sum(doc_start[idxs])

if ndocs > s:
    idxs = idxs[:np.argwhere(np.cumsum(doc_start[idxs]) == s)[0][0]]
elif ndocs < s:  # not enough validation data
    moreidxs = np.argwhere(gt != -1)[:, 0]
    deficit = s - ndocs
    ndocs = np.sum(doc_start[moreidxs])
    if ndocs > deficit:
        moreidxs = moreidxs[:np.argwhere(np.cumsum(doc_start[moreidxs]) == deficit)[0][0]]
    idxs = np.concatenate((idxs, moreidxs))

annos_val = annos[idxs]
doc_start_val = doc_start[idxs]
features_val = features[idxs]
gt_val = gt_val[idxs]

output_dir = os.path.join(evaluation.experiment.output_root_dir, 'ner')
exp = Experiment(output_dir, 9, annos, gt, doc_start, features, annos_val, gt_val, doc_start_val, features_val,
                 max_iter=20)
beta_factors = [0.1, 1, 10]
diags = [0.1, 1, 10, 100]
factors = [0.1, 1, 10]
methods_to_tune = [
    # 'ibcc',
    # 'HMM_crowd',
    'bsc_acc_integrateIF',
    'bsc_spam_integrateIF',
    'bsc_cv_integrateIF',
    'bsc_cm_integrateIF',
    'bsc_seq_integrateIF',
    'bsc_cm_integrateIF_noHMM',
    'bsc_seq_integrateIF_noHMM',
    'bsc_ibcc',
    'bsc_seq',
]
#
for m, method in enumerate(methods_to_tune):
    print('TUNING %s' % method)

    best_scores = exp.tune_alpha0(diags, factors, beta_factors, method, metric_idx_to_optimise=8)

    best_idxs = best_scores[1:].astype(int)
    exp.beta0_factor = beta_factors[best_idxs[0]]
    exp.alpha0_diags = diags[best_idxs[1]]
    exp.alpha0_factor = factors[best_idxs[2]]

    print('Best values: %f, %f, %f' % (exp.beta0_factor, exp.alpha0_diags, exp.alpha0_factor))

    # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
    exp.methods = [method]
    exp.run_methods(new_data=regen_data)

