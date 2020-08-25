'''

@author: Edwin Simpson
'''
import os

import evaluation.experiment
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

regen_data = False
gt, annos, doc_start, features, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_val, _ = \
    load_data.load_ner_data(regen_data)

# -------------------- debug or tune with subset -------
# s = 500
# idxs = np.argwhere(gt != -1)[:, 0] # for testing
# # idxs = np.argwhere(gt_val != -1)[:, 0] # for tuning
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

# -------------------------------------------------------------------------------------
beta0_factor = 1
alpha0_diags = 10
alpha0_factor = 10
output_dir = os.path.join(evaluation.experiment.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags,
                                                                                    alpha0_factor))
exp = Experiment(output_dir, 9, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
                 alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
exp.methods = [
                # 'bac_seq_integrateIF',
                'bac_seq_integrateIF_thenLSTM',
                'bac_seq_integrateIF_integrateLSTM_atEnd',
]

# should run both task 1 and 2.
exp.run_methods(new_data=regen_data)

# -------------------------------------------------------------------------------------

beta0_factor = 0.1
alpha0_diags = 0.1
alpha0_factor = 0.1
output_dir = os.path.join(evaluation.experiment.output_root_dir, 'ner3_%f_%f_%f' % (beta0_factor, alpha0_diags,
                                                                                    alpha0_factor))
exp = Experiment(output_dir, 9, annos, gt, doc_start, features, annos, gt_val, doc_start, features,
                 alpha0_factor=alpha0_factor, alpha0_diags=alpha0_diags, beta0_factor=beta0_factor, max_iter=20)
# run all the methods that don't require tuning here
exp.methods = [
                'gt_thenLSTM', # train the LSTM on the real gold labels
                'HMM_crowd_thenLSTM',
]

# should run both task 1 and 2.
exp.run_methods(new_data=regen_data)
