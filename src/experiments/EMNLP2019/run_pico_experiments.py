'''
Created on April 27, 2018

@author: Edwin Simpson
'''
import os

import evaluation.experiment
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np


regen_data = False
gt, annos, doc_start, features, gt_val, _, _, _ = load_data.load_biomedical_data(regen_data)
#  , debug_subset_size=1000) # include this argument to debug with small dataset

# ------------------------------------------------------------------------------------------------
#
# # only hmm_Crowd actually uses these hyperparameters
# beta0_factor = 0.1
# alpha0_diags = 0.1
# alpha0_factor = 0.1
# output_dir = os.path.join(evaluation.experiment.output_root_dir, 'pico3')
# exp = Experiment(output_dir, 3, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
#                  alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
# # # run all the methods that don't require tuning here
# exp.methods = [
#     'best',
#     'worst',
#     'majority',
#     'ds',
#     'mace',
#     'HMM_crowd',
# ]
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(new_data=regen_data)

# ------------------------------------------------------------------------------------------------

beta0_factor = 0.1
alpha0_diags = 10
alpha0_factor = 10
output_dir = os.path.join(evaluation.experiment.output_root_dir, 'pico3_%f_%f_%f' % (beta0_factor, alpha0_diags,
                                                                                     alpha0_factor))
exp = Experiment(output_dir, 3, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
                 alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
# # run all the methods that don't require tuning here
exp.methods = [
    'ibcc',
]
# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(new_data=regen_data)

# # ------------------------------------------------------------------------------------------------
#
# beta0_factor = ?
# alpha0_diags = ?
# alpha0_factor = ?
# output_dir = os.path.join(evaluation.experiment.output_root_dir, 'pico3_%f_%f_%f' % (beta0_factor, alpha0_diags,
#                                                                                      alpha0_factor))
# exp = Experiment(output_dir, 3, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
#                  alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor,
#                  max_iter=20)
# # run all the methods that don't require tuning here
# exp.methods = [
#                 'bsc_acc_integrateIF',
# ]
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(new_data=regen_data)

# # ------------------------------------------------------------------------------------------------
#
# beta0_factor = ?
# alpha0_diags = ?
# alpha0_factor = ?
# output_dir = os.path.join(evaluation.experiment.output_root_dir, 'pico3_%f_%f_%f' % (beta0_factor, alpha0_diags,
#                                                                                      alpha0_factor))
# exp = Experiment(output_dir, 3, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
#                  alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor,
#                  max_iter=20)
# # run all the methods that don't require tuning here
# exp.methods = [
#                 'bsc_spam_integrateIF',
# ]
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(new_data=regen_data)

# ------------------------------------------------------------------------------------------------

beta0_factor = 0.1  # 0.01
alpha0_diags = 1.0  # 0.1
alpha0_factor = 0.1  # 0.1
output_dir = os.path.join(evaluation.experiment.output_root_dir, 'pico3_%f_%f_%f' % (beta0_factor, alpha0_diags,
                                                                                     alpha0_factor))
exp = Experiment(output_dir, 3, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
                 alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor,
                 max_iter=20)
# run all the methods that don't require tuning here
exp.methods = [
                'bsc_cv_integrateIF',
]
# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(new_data=regen_data)

# # ------------------------------------------------------------------------------------------------
#
# beta0_factor = 0.01
# alpha0_diags = 1.0
# alpha0_factor = 10.0
# output_dir = os.path.join(evaluation.experiment.output_root_dir, 'pico3_%f_%f_%f' % (beta0_factor, alpha0_diags,
#                                                                                      alpha0_factor))
# exp = Experiment(output_dir, 3, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
#                  alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor,
#                  max_iter=20)
# # run all the methods that don't require tuning here
# exp.methods = [
#                 'bsc_cm_integrateIF',
# ]
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(new_data=regen_data)

# ------------------------------------------------------------------------------------------------

beta0_factor = 1
alpha0_diags = 10
alpha0_factor = 10
best_begin_factor = 10
output_dir = os.path.join(evaluation.experiment.output_root_dir, 'pico3_%f_%f_%f' % (beta0_factor, alpha0_diags,
                                                                                     alpha0_factor))
exp = Experiment(output_dir, 3, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
                 alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor,
                 max_iter=20, begin_factor=best_begin_factor)
# run all the methods that don't require tuning here
exp.methods = [
                'bac_seq_integrateIF',
]
# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(new_data=regen_data)

# ------------------------------------------------------------------------------------------------
# tune with small dataset to save time
s = 300
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

tune_annos = annos[idxs]
tune_doc_start = doc_start[idxs]
tune_text = features[idxs]
gt_val = gt_val[idxs]

beta_factors = [0.1, 1]
diags = [0.1, 1, 10]
factors = [0.1, 1, 10]

methods_to_tune = [
    'ibcc',
    'bac_acc_integrateIF',
    'bac_mace_integrateIF',
    # 'bac_vec_integrateIF',
    # 'bac_ibcc_integrateIF',
    'bac_ibcc_integrateIF_noHMM',
    'bac_seq_integrateIF_noHMM',
    'bac_ibcc',
    'bac_seq',
]
output_dir = os.path.join(evaluation.experiment.output_root_dir, 'pico')
exp = Experiment(output_dir, 3, annos, gt, doc_start, features, tune_annos, gt_val, tune_doc_start, tune_text,
                 max_iter=20, begin_factor=10)

for m, method in enumerate(methods_to_tune):
    print('TUNING %s' % method)

    best_scores = exp.tune_alpha0(diags, factors, beta_factors, method, metric_idx_to_optimise=11)
    best_idxs = best_scores[1:].astype(int)

    exp.beta0_factor = beta_factors[best_idxs[0]]
    exp.alpha0_diags = diags[best_idxs[1]]
    exp.alpha0_factor = factors[best_idxs[2]]

    print('Best values: %f, %f, %f' % (exp.beta0_factor, exp.alpha0_diags, exp.alpha0_factor))

    # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
    exp.methods = [method]
    exp.run_methods(new_data=regen_data)
