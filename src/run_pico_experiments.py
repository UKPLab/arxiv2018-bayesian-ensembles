'''
Created on April 27, 2018

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_annotator_combination/output/bio_task1/'

gt, annos, doc_start, text, gt_dev, doc_start_dev, text_dev = load_data.load_biomedical_data(True)

exp = Experiment(None, 3, annos.shape[1], None)
exp.methods = ['majority', 'HMM_crowd_then_LSTM', 'best', 'worst', 'ibcc', 'bac_mace', 'bac_acc', 'bac_seq', 'bac_ibcc']#, 'majority']#['mace', 'best', 'worst', 'majority']#'bac', 'ibcc', 'mace', 'majority'] # 'bac', 'clustering', 'ibcc', 'mace',

exp.save_results = True
exp.opt_hyper = False #True

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text,
                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev)

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