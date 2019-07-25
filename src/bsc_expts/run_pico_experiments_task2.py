'''
Created on April 27, 2018

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_sequence_combination/output/pico_task2/'

regen_data = False
gt, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev = \
    load_data.load_biomedical_data(regen_data)

# TASK 2 also needs to reload the optimised hyperparameters.

gold_labelled = gt.flatten() != -1
crowd_labelled = np.invert(gold_labelled)

annos_tr = annos[crowd_labelled, :]
gt_tr = gt[crowd_labelled] # for evaluating performance on labelled data -- these are all -1 so we can ignore these results
doc_start_tr = doc_start[crowd_labelled]
text_tr = text[crowd_labelled]

gt_test = gt[gold_labelled]
doc_start_test = doc_start[gold_labelled]
text_test = text[gold_labelled]

best_bac_wm = 'bac_seq' # choose model with best score for the different BAC worker models
best_nu0factor = 100
best_diags = 1
best_factor = 1

nu_factors = [0.1, 10, 100]
diags = [0.1, 1, 10, 100] #, 50, 100]#[1, 50, 100]#[1, 5, 10, 50]
factors = [0.1, 1, 10]

exp = Experiment(None, 3, annos.shape[1], None, max_iter=20)

exp.save_results = True
exp.opt_hyper = False #True

exp.nu0_factor = best_nu0factor
exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor

exp.alpha0_diags_lstm = 0.1
exp.alpha0_factor_lstm = 0.1

exp.methods =  [
                # best_bac_wm + '_integrateIF_integrateLSTM_atEnd',
                # best_bac_wm + '_integrateIF_then_LSTM',
                'gt_then_LSTM',
                'HMM_crowd_then_LSTM',
                ]

exp.run_methods(annos_tr, gt_tr, doc_start_tr, output_dir, text_tr,
                ground_truth_nocrowd=gt_test, doc_start_nocrowd=doc_start_test, text_nocrowd=text_test,
                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                new_data=False)
