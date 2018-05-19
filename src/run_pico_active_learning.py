'''
Created on April 27, 2018

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

gt, annos, doc_start, text, gt_dev, doc_start_dev, text_dev = load_data.load_biomedical_data(False)

exp = Experiment(None, 3, annos.shape[1], None)

exp.save_results = True
exp.opt_hyper = False #True

exp.alpha0_diags = 100
exp.alpha0_factor = 1

# ACITVE LEARNING WITH TASK 2 also needs to reload the optimised hyperparameters.

# as task 2 -- train on crowdsourced data points with no gold labels,
# then test on gold-labelled data without using crowd labels
output_dir = '../../data/bayesian_annotator_combination/output/bio_al/'

gold_labelled = gt.flatten() != -1
crowd_labelled = np.invert(gold_labelled)

# for debugging
# crowd_labelled[np.random.choice(len(crowd_labelled), int(np.floor(len(crowd_labelled) * 0.99)), replace=False), 0] = False
# crowd_labelled = crowd_labelled[:, 0]
#
# gold_labelled[np.random.choice(len(gold_labelled), int(np.floor(len(gold_labelled) * 0.99)), replace=False), 0] = False
# gold_labelled = gold_labelled[:, 0]
#
# gt_dev = gt_dev[:100]
# doc_start_dev = doc_start_dev[:100]
# text_dev = text_dev[:100]

annos_tr = annos[crowd_labelled, :]
gt_tr = gt[crowd_labelled] # for evaluating performance on labelled data -- these are all -1 so we can ignore these results
doc_start_tr = doc_start[crowd_labelled]
text_tr = text[crowd_labelled]

gt_test = gt[gold_labelled]
doc_start_test = doc_start[gold_labelled]
text_test = text[gold_labelled]

exp = Experiment(None, 3, annos.shape[1], None)
exp.methods = ['bac_ibcc_integrateLSTM',
               'HMM_crowd_then_LSTM']

exp.save_results = True
exp.opt_hyper = False #True

exp.run_methods(annos_tr, gt_tr, doc_start_tr, output_dir, text_tr,
                ground_truth_nocrowd=gt_test, doc_start_nocrowd=doc_start_test, text_nocrowd=text_test,
                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                active_learning=True)