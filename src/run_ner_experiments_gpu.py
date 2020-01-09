'''

@author: Edwin Simpson
'''
import os

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = os.path.join(load_data.output_root_dir, 'ner')

regen_data = False
gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val, gt_all = \
    load_data.load_ner_data(regen_data)

# Defaults ---------

niter = 20

best_nu0factor = 1
best_diags = 10
best_factor = 10
best_acc_bias = 0

# best_bac_wm = 'bac_seq' #'unknown' # choose model with best score for the different BAC worker models
# best_bac_wm_score = -np.inf

# The following code tunes on dev set: -----------------------
#
# exp = Experiment(None, 9, annos.shape[1], None, max_iter=niter)
# exp.save_results = True
# exp.opt_hyper = False#True
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
# exp.alpha0_acc_bias = best_acc_bias
#
# nu_factors = [0.1]
# lstm_diags = [1, 10, 100, 1000]
# lstm_factors = [0.1, 100]
# # acc_biases = [1, 10, 100]
#
# methods_to_tune = [
#                    'bac_seq_integrateIF_integrateLSTM_atEnd',
#                    ]
# # in this case, tune with the full dataset because LSTM doesn't work otherwise
#
# for m, method in enumerate(methods_to_tune):
#     print('TUNING %s' % method)
#
#     exp.bac_iterative_learning = True  # don't reset BAC each iteration as we only need to change the LSTM component.
#
#     best_scores = exp.tune_alpha0(lstm_diags,
#                                   lstm_factors,
#                                   nu_factors,
#                                   method,
#                                   annos,
#                                   gt_task1_val,
#                                   doc_start,
#                                   output_dir,
#                                   text,
#                                   tune_lstm=True,
#                                   ground_truth_val=gt_val,
#                                   doc_start_val=doc_start_val,
#                                   text_val=text_val)
#
#     best_idxs = best_scores[1:].astype(int)
#     exp.alpha0_diags_lstm = lstm_diags[best_idxs[1]]
#     exp.alpha0_factor_lstm = lstm_factors[best_idxs[2]]
#
#     print('Best values: %f, %f, %f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
#
#     exp.methods = [method]
#     exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, return_model=False,
#                 ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#                 ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd,
#                 text_nocrowd=text_nocrowd,
#                 new_data=regen_data
#                 )
#
#     best_score = best_scores[0]
#     if 'bac_seq' in method and best_score > best_bac_wm_score:
#         best_bac_wm = 'bac_' + method.split('_')[1]
#         best_bac_wm_score = best_score
#         best_diags = exp.alpha0_diags
#         best_factor = exp.alpha0_factor
#         best_nu0factor = exp.nu0_factor
#         best_acc_bias = exp.alpha0_acc_bias
#     #
#     # if 'bac_seq' in method and best_score > best_bac_wm_score:
#     #     best_bac_wm_score = best_score
#     #     best_acc_bias = exp.alpha0_acc_bias
#     #     best_bac_wm = 'bac_' + method.split('_')[1]
#
# print('best BAC method tested here = %s' % best_bac_wm)

# --------------------
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
#                 best_bac_wm + '_integrateIF_then_LSTM',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(
#     annos, gt, doc_start, output_dir, text,
#     ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#     ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
#     new_data=regen_data
# )

# # reset to free memory? ------------------------------------------------------------------------------------------------
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

# # reset ------------------------------------------------------------------------------------------------
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
#                 'gt_then_LSTM',
# ]
#
# # should run both task 1 and 2.
#
# exp.run_methods(
#     annos, gt, doc_start, output_dir, text,
#     ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#     ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
#     new_data=regen_data, ground_truth_all_points=gt_all
# )

# # -------------------- debug with subset -------
# s = 1000
# idxs = np.argwhere(gt!=-1)[:s, 0]
# gt = gt[idxs]
# annos = annos[idxs]
# doc_start = doc_start[idxs]
# text = text[idxs]
# gt_task1_val = gt_task1_val[idxs]
# # -------------------------

# niter = 20 # variational inference iterations
#
# exp = Experiment(None, 9, annos.shape[1], None, max_iter=niter)
# exp.save_results = True
# exp.opt_hyper = False#True
#
# exp.nu0_factor = 10
# exp.alpha0_diags = 1
# exp.alpha0_factor = 1
# output_dir = os.path.join(load_data.output_root_dir, 'ner_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
# # bsc_seq has got worse: maybe due to the changed constraints? Beta OO preference might make less sense for NER
# # because the entities are much more frequent. Solution: use different settings for NER rather than try to find a single setup.
# # Or because we modified the prior by taking out the /2 bit.
# # Or because we changed the nu0 prior for the word distributions.
# exp.methods =  [
#                 # 'majority',
#                 # 'ibcc',
#                 'bac_vec_integrateIF',
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

#----------------------------------------------------------------------------
# niter = 20 # variational inference iterations
#
# exp = Experiment(None, 9, annos.shape[1], None, max_iter=niter)
# exp.save_results = True
# exp.opt_hyper = False#True
#
# exp.nu0_factor = 0.1
# exp.alpha0_diags = 100
# exp.alpha0_factor = 0.1
# output_dir = os.path.join(load_data.output_root_dir, 'ner_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
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

#-------------------------------------------------------------------------------------
niter = 20 # variational inference iterations

exp = Experiment(None, 9, annos.shape[1], None, max_iter=niter)
exp.save_results = True
exp.opt_hyper = False#True

exp.nu0_factor = 0.1
exp.alpha0_diags = 1
exp.alpha0_factor = 1
# exp.begin_factor = 10
output_dir = os.path.join(load_data.output_root_dir, 'ner2_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
exp.methods =  [ # acc seems to be broken; seq has a minor error somewhere; HMM has bigger error, probably in the lower bound qt computation
                'bac_seq_integrateIF',
                # 'bac_seq_integrateIF',
                # best_bac_wm + '_integrateIF_integrateLSTM_atEnd',
                # best_bac_wm + '_integrateIF_then_LSTM',
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

exp.nu0_factor = 1
exp.alpha0_diags = 10
exp.alpha0_factor = 10
# exp.begin_factor = 10
output_dir = os.path.join(load_data.output_root_dir, 'ner2_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
exp.methods =  [ # acc seems to be broken; seq has a minor error somewhere; HMM has bigger error, probably in the lower bound qt computation
                'bac_seq_integrateIF',
                # 'bac_seq_integrateIF',
                # best_bac_wm + '_integrateIF_integrateLSTM_atEnd',
                # best_bac_wm + '_integrateIF_then_LSTM',
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

exp.nu0_factor = 1
exp.alpha0_diags = 1
exp.alpha0_factor = 1
output_dir = os.path.join(load_data.output_root_dir, 'ner2_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
exp.methods =  [
                'bac_seq_integrateIF',
                # best_bac_wm + '_integrateIF_integrateLSTM_atEnd',
                # best_bac_wm + '_integrateIF_then_LSTM',
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

exp.nu0_factor = 1
exp.alpha0_diags = 1
exp.alpha0_factor = 10
output_dir = os.path.join(load_data.output_root_dir, 'ner2_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
exp.methods =  [
                'bac_seq_integrateIF',
                # best_bac_wm + '_integrateIF_integrateLSTM_atEnd',
                # best_bac_wm + '_integrateIF_then_LSTM',
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

exp.nu0_factor = 10
exp.alpha0_diags = 10
exp.alpha0_factor = 10
output_dir = os.path.join(load_data.output_root_dir, 'ner2_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
exp.methods =  [
                'bac_seq_integrateIF',
                # best_bac_wm + '_integrateIF_integrateLSTM_atEnd',
                # best_bac_wm + '_integrateIF_then_LSTM',
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

exp.nu0_factor = 0.1
exp.alpha0_diags = 10
exp.alpha0_factor = 10
output_dir = os.path.join(load_data.output_root_dir, 'ner2_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
exp.methods =  [
                'bac_seq_integrateIF',
                # best_bac_wm + '_integrateIF_integrateLSTM_atEnd',
                # best_bac_wm + '_integrateIF_then_LSTM',
]

# should run both task 1 and 2.
exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    # ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    ground_truth_nocrowd=None, doc_start_nocrowd=None, text_nocrowd=None,
    new_data=regen_data
)