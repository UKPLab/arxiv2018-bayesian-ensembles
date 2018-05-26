'''
Created on April 27, 2018

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_annotator_combination/output/bio_task1/'

# Todo turn on SENTENCE SPLITTING!!!
# Todo what happened to the span-level prec/recall scores? See last result on Barney for PICO.
# Todo Make it save the models for BAC

gt, annos, doc_start, text, gt_dev, doc_start_dev, text_dev = load_data.load_biomedical_data(False)

exp = Experiment(None, 3, annos.shape[1], None)

exp.save_results = True
exp.opt_hyper = False #True

exp.alpha0_diags = 100
exp.alpha0_factor = 9

diags = [0.1, 1, 50, 100]#[1, 50, 100]#[1, 5, 10, 50]
factors = [0.1, 1, 9, 36]#[36, 49, 64]#[1, 4, 9, 16, 25]
methods_to_tune = ['ibcc',
                   'bac_vec_integrateBOF',
                   'bac_seq_integrateBOF',
                   'bac_ibcc_integrateBOF',
                   'bac_acc_integrateBOF',
                   'bac_mace_integrateBOF'
                   ]

best_bac_wm = 'bac_seq' # choose model with best score for the different BAC worker models
best_bac_wm_score = -np.inf

# tune with small dataset to save time
idxs = np.argwhere(gt_dev != -1)[:, 0]

tune_gt_dev = gt_dev[idxs]
tune_gt = gt[idxs]
tune_annos = annos[idxs]
tune_doc_start = doc_start[idxs]
tune_text = text[idxs]

for m, method in enumerate(methods_to_tune):
    print('TUNING %s' % method)

    best_scores = exp.tune_alpha0(diags, factors, method, tune_annos, tune_gt_dev, tune_doc_start,
                                  output_dir, tune_text, metric_idx_to_optimise=11)
    best_idxs = best_scores[1:].astype(int)
    exp.alpha0_diags = diags[best_idxs[0]]
    exp.alpha0_factor = factors[best_idxs[1]]

    print('Best values: %f, %f' % (exp.alpha0_diags, exp.alpha0_factor))

    # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
    exp.methods = [method]
    exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True,
                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev)

    best_score = np.max(best_scores)
    if 'bac' in method and best_score > best_bac_wm_score:
        best_bac_wm = method
        best_bac_wm_score = best_score
        best_diags = exp.alpha0_diags
        best_factor = exp.alpha0_factor

print('best BAC method = %s' % best_bac_wm)

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor

# exp.alpha0_diags = 1#50
# exp.alpha0_factor = 1#9

# run all the methods that don't require tuning here
exp.methods =  ['majority',
                #'ibcc'
                #'best',
                #'worst',
                'HMM_crowd',
                best_bac_wm,
                best_bac_wm + '_integrateBOF_then_LSTM',
                best_bac_wm + '_integrateBOF_integrateLSTM_atEnd',
                best_bac_wm + '_integrateLSTM_integrateBOF_atEnd_noHMM',
                'HMM_crowd_then_LSTM',
]

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text,
                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev)