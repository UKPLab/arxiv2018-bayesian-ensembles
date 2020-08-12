'''
Created on April 27, 2018

@author: Edwin Simpson
'''
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np
import os

gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_val, _ = \
    load_data.load_ner_data(False)

# # debug with subset -------
# s = 100
# idxs = np.argwhere(gt!=-1)[:s, 0]
# gt = gt[idxs]
# annos = annos[idxs]
# doc_start = doc_start[idxs]
# text = text[idxs]
# gt_val = gt_val[idxs]
# # -------------------------

num_reps = 10
batch_frac = 0.03
AL_iters = 10

output_dir = os.path.join(load_data.output_root_dir, 'ner_al')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# ACTIVE LEARNING WITH UNCERTAINTY SAMPLING
for rep in range(1, num_reps):

    beta0_factor = 0.1
    alpha0_diags = 1  # best_diags
    alpha0_factor = 1  # 9 # best_factor
    exp = Experiment(output_dir, 9, annos, gt, doc_start, text, annos, gt_val, doc_start, text,
                     alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor,
                     max_iter=20, crf_probs=True, rep=rep)
    exp.methods = [
        'bac_seq_integrateIF',
        'HMM_crowd',
    ]
    results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(
        active_learning=True, AL_batch_fraction=batch_frac, max_AL_iters=AL_iters
    )

    beta0_factor = 0.1
    alpha0_diags = 100 # best_diags
    alpha0_factor = 0.1 #9 # best_factor
    exp = Experiment(output_dir, 9, annos, gt, doc_start, text, annos, gt_val, doc_start, text,
                     alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor,
                     max_iter=20, crf_probs=True, rep=rep)
    # run all the methods that don't require tuning here
    exp.methods =  [
        'bac_ibcc_integrateIF',
    ]

    results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(
                        active_learning=True, AL_batch_fraction=batch_frac, max_AL_iters=AL_iters
    )

    beta0_factor = 10
    alpha0_diags = 1 # best_diags
    alpha0_factor = 1#9 # best_factor
    exp = Experiment(output_dir, 9, annos, gt, doc_start, text, annos, gt_val, doc_start, text,
                     alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor,
                     max_iter=20, crf_probs=True, rep=rep)
    exp.methods =  [
        'bac_vec_integrateIF',
    ]

    results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(
                        active_learning=True, AL_batch_fraction=batch_frac, max_AL_iters=AL_iters
    )

    beta0_factor = 0.1
    alpha0_diags = 1 # best_diags
    alpha0_factor = 0.1#9 # best_factor
    exp = Experiment(output_dir, 9, annos, gt, doc_start, text, annos, gt_val, doc_start, text,
                     alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor,
                     max_iter=20, crf_probs=True, rep=rep)
    exp.methods =  [
        'ibcc',
        'ds',
        'majority'
    ]

    results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(
                        active_learning=True, AL_batch_fraction=batch_frac, max_AL_iters=AL_iters
    )
