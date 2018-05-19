'''
Created on April 27, 2018

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_annotator_combination/output/bio_task1/'

gt, annos, doc_start, text, gt_dev, doc_start_dev, text_dev = load_data.load_biomedical_data(False)

exp = Experiment(None, 3, annos.shape[1], None)

exp.save_results = True
exp.opt_hyper = False #True

exp.alpha0_diags = 100
exp.alpha0_factor = 9

diags = [1, 50, 100]#[1, 50, 100]#[1, 5, 10, 50]
factors = [1, 4, 9, 36]

methods_to_tune = ['ibcc', 'bac_acc', 'bac_ibcc', 'bac_seq', 'bac_mace']

# best_bac_wm = 'bac_ibcc' # choose model with best score for the different BAC worker models
# best_bac_wm_score = -np.inf
#
# # tune with small dataset to save time
# s = 250
# idxs = np.argwhere(gt_dev != -1)[:, 0]
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
# tune_gt_dev = gt_dev[idxs]
#
# for m, method in enumerate(methods_to_tune):
#     print('TUNING %s' % method)
#
#     best_scores = exp.tune_alpha0(diags, factors, method, tune_annos, tune_gt_dev, tune_doc_start,
#                                   output_dir, tune_text)
#     best_idxs = best_scores[1:].astype(int)
#     exp.alpha0_diags = diags[best_idxs[0]]
#     exp.alpha0_factor = factors[best_idxs[1]]
#
#     print('Best values: %f, %f' % (exp.alpha0_diags, exp.alpha0_factor))
#
#     # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
#     exp.methods = [method]
#     exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True,
#                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev)
#
#     best_score = np.max(best_scores)
#     if 'bac' in method and best_score > best_bac_wm_score:
#         best_bac_wm = method
#         best_bac_wm_score = best_score
#         best_diags = exp.alpha0_diags
#         best_factor = exp.alpha0_factor
#
# print('best BAC method = %s' % best_bac_wm)

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
#
# exp.alpha0_diags = 50
# exp.alpha0_factor = 9

# run all the methods that don't require tuning here
exp.methods =  ['majority',
                'best',
                'worst',
                'HMM_crowd',
                'HMM_crowd_then_LSTM',
                best_bac_wm + '_then_LSTM',
                best_bac_wm + '_integrateLSTM',
                # 'bac_acc' + '_integrateLSTM'
                ]

if best_bac_wm != 'bac_acc':
    exp.methods.append('bac_acc_integrateLSTM')

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text,
                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev)

# TASK 2 also needs to reload the optimised hyperparameters.

# For now we are just running task 1 so we can see which variant of BAC works best before running it with LSTM. Then we
# will change the methods selected below and uncomment the code, and comment out the code above.

# # now for task 2 -- train on crowdsourced data points with no gold labels, then test on gold-labelled data without using crowd labels
# output_dir = '../../data/bayesian_annotator_combination/output/bio_task2/'
#
# gold_labelled = gt != -1
# crowd_labelled = np.invert(gold_labelled)
#
# annos_tr = annos[crowd_labelled, :]
# gt_tr = gt[crowd_labelled] # for evaluating performance on labelled data -- these are all -1 so we can ignore these results
# doc_start_tr = doc_start[crowd_labelled]
# text_tr = text[crowd_labelled]
#
# gt_test = gt[gold_labelled]
# doc_start_test = doc_start[gold_labelled]
# text_test = text[gold_labelled]
#
# exp = Experiment(None, 3, annos.shape[1], None)
# exp.methods = ['bac_acc_then_LSTM', 'HMM_crowd_then_LSTM']#, 'bac_mace', 'bac_acc', 'bac_seq', 'bac_ibcc', 'ibcc', 'best', 'worst', 'majority']#['mace', 'best', 'worst', 'majority']#'bac', 'ibcc', 'mace', 'majority'] # 'bac', 'clustering', 'ibcc', 'mace',
#
# exp.save_results = True
# exp.opt_hyper = False #True
#
# exp.run_methods(annos_tr, gt_tr, doc_start_tr, output_dir, text_tr, gt_test, doc_start_test, text_test,
#                   ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev)