'''
Created on April 27, 2018

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_annotator_combination/output/bio_task2/'

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


exp = Experiment(None, 3, annos.shape[1], None, max_iter=20)

exp.save_results = True
exp.opt_hyper = False #True

exp.nu0_factor = 0.1
exp.alpha0_diags = 0.1
exp.alpha0_factor = 0.1

# run all the methods that don't require tuning here
exp.methods =  [
                'HMM_crowd',
                'HMM_crowd_then_LSTM',
                ]

exp.run_methods(annos_tr, gt_tr, doc_start_tr, output_dir, text_tr,
                ground_truth_nocrowd=gt_test, doc_start_nocrowd=doc_start_test, text_nocrowd=text_test,
                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev)

exp.methods =  [
                'bac_seq_integrateBOF_integrateLSTM_atEnd',
                ]

exp.run_methods(annos_tr, gt_tr, doc_start_tr, output_dir, text_tr,
                ground_truth_nocrowd=gt_test, doc_start_nocrowd=doc_start_test, text_nocrowd=text_test,
                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev)

exp.methods =  [
                'bac_seq_integrateBOF_then_LSTM',
                ]

exp.run_methods(annos_tr, gt_tr, doc_start_tr, output_dir, text_tr,
                ground_truth_nocrowd=gt_test, doc_start_nocrowd=doc_start_test, text_nocrowd=text_test,
                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev)