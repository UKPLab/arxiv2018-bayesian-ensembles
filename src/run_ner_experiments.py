'''
Created on April 27, 2018

@author: Edwin Simpson
'''
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np
# import pandas as pd
# from data.load_data import _map_ner_str_to_labels

output_dir = '../../data/bayesian_annotator_combination/output/ner-by-sentence/'

regen_data = True
gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val = \
    load_data.load_ner_data(regen_data)
#
# datapath = '../../data/bayesian_annotator_combination/data/ner-mturk'
#
# annos = pd.read_csv(datapath + '/answers.txt', skip_blank_lines=False, na_filter=False, delim_whitespace=True,
#                     header=None)
# doc_gaps = (annos.ix[:, 0] == '') & (annos.ix[:, 1] == '')
# doc_content = np.invert(doc_gaps)
#
# doc_start = np.zeros(len(doc_gaps))
# doc_start[doc_gaps[doc_gaps].index[:-1] + 1] = 1
# doc_start = doc_start[doc_content]
# doc_start[0] = 1
#
# annos = annos.ix[doc_content, 1:].values
# annos = _map_ner_str_to_labels(annos)
#
# ground_truth = pd.read_csv(datapath + '/ground_truth.txt', skip_blank_lines=True, na_filter=False,
#                            delim_whitespace=True, header=None)
#
# text = ground_truth.ix[:, 0].values
# gt = ground_truth.ix[:, 1].values
# gt = _map_ner_str_to_labels(gt)
#
# # split crowd-labelled data into val and train
# ntrain = int(len(gt) * 0.5)
#
# gt_task1_val = np.copy(gt)
#
# gt[ntrain:] = -1
# gt_task1_val[:ntrain] = -1
#
# test_ground_truth = pd.read_csv(datapath + '/testset.txt', skip_blank_lines=True, na_filter=False,
#                            delim_whitespace=True, header=None)
#
# doc_gaps = (test_ground_truth.ix[:, 0] == '') & (test_ground_truth.ix[:, 1] == '')
# doc_content = np.invert(doc_gaps)
#
# doc_start_nocrowd = np.zeros(len(doc_gaps))
# doc_start_nocrowd[doc_gaps[doc_gaps].index[:-1] + 1] = 1
# doc_start_nocrowd[doc_content]
# doc_start_nocrowd[0] = 1
#
# test_ground_truth = test_ground_truth[doc_content]
#
# text_nocrowd = test_ground_truth.ix[:, 0].values
# gt_nocrowd = test_ground_truth.ix[:, 1].values
# gt_nocrowd = _map_ner_str_to_labels(gt_nocrowd)
#
# print('loading text data for task 2 val')
# savepath = '../../data/bayesian_annotator_combination/data/ner/'  # location to save our csv files to
# text_val = pd.read_csv(savepath + '/task2_val_text.csv', skip_blank_lines=False, header=None)
# text_val = text_val.fillna(' ').values
#
# print('loading doc_starts for task 2 val')
# doc_start_val = pd.read_csv(savepath + '/task2_val_doc_start.csv', skip_blank_lines=False, header=None).values
#
# print('loading ground truth for task 2 val')
# gt_val = pd.read_csv(savepath + '/task2_val_gt.csv', skip_blank_lines=False, header=None).values

# debug with subset -------
# s = 1000
# idxs = np.argwhere(gt!=-1)[:s, 0]
# gt = gt[idxs]
# annos = annos[idxs]
# doc_start = doc_start[idxs]
# print('No. documents:')
# print(np.sum(doc_start))
# text = text[idxs]
# gt_task1_val = gt_task1_val[idxs]
# -------------------------

exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1)
exp.save_results = True
exp.opt_hyper = False#True

# best IBCC setting so far is diag=100, factor=36. Let's reuse this for BIO and all BAC_IBCC runs.

diags = [0.1, 1, 50, 100]#[1, 50, 100]#[1, 5, 10, 50]
factors = [0.1, 1, 9, 36]#[36, 49, 64]#[1, 4, 9, 16, 25]
methods_to_tune = ['ibcc', 'bac_vec', 'bac_seq', 'bac_ibcc', 'bac_acc', 'bac_mace']

best_bac_wm = 'bac_vec' #'unknown' # choose model with best score for the different BAC worker models
best_bac_wm_score = -np.inf

# tune with small dataset to save time
s = 250
idxs = np.argwhere(gt_task1_val != -1)[:, 0]
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

tune_gt = gt[idxs]
tune_annos = annos[idxs]
tune_doc_start = doc_start[idxs]
tune_text = text[idxs]
tune_gt_task1_val = gt_task1_val[idxs]

for m, method in enumerate(methods_to_tune):
    print('TUNING %s' % method)

    best_scores = exp.tune_alpha0(diags, factors, method, tune_annos, tune_gt_task1_val, tune_doc_start,
                                  output_dir, tune_text)
    best_idxs = best_scores[1:].astype(int)
    exp.alpha0_diags = diags[best_idxs[0]]
    exp.alpha0_factor = factors[best_idxs[1]]

    print('Best values: %f, %f' % (exp.alpha0_diags, exp.alpha0_factor))

    # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
    exp.methods = [method]
    exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True,
                ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val)

    best_score = np.max(best_scores)
    if 'bac' in method and best_score > best_bac_wm_score:
        best_bac_wm = method
        best_bac_wm_score = best_score
        best_diags = exp.alpha0_diags
        best_factor = exp.alpha0_factor

print('best BAC method tested here = %s' % best_bac_wm)
#
exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor

# exp.alpha0_diags = 50
# exp.alpha0_factor = 1

# run all the methods that don't require tuning here
exp.methods =  [
                #'majority',
                #'mace',
                #'ibcc',
                'ds',
                #'best', 'worst',
                #'HMM_crowd',
                best_bac_wm,
                best_bac_wm + 'integrateBOF_then_LSTM',
                best_bac_wm + '_integrateBOF_integrateLSTM_atEnd',
                best_bac_wm + '_integrateLSTM_integrateBOF_atEnd_noHMM',
                'HMM_crowd_then_LSTM',
]

# should run both task 1 and 2.

results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    new_data=regen_data
)