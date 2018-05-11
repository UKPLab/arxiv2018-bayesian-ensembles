'''
Created on April 27, 2018

@author: Edwin Simpson
'''
from evaluation.experiment import Experiment
import data.load_data as load_data

output_dir = '../../data/bayesian_annotator_combination/output/ner/'

gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_val, doc_start_val, text_val = \
    load_data.load_ner_data(True)

exp = Experiment(None, 9, annos.shape[1], None)
exp.methods = ['majority', 'best', 'worst', 'ibcc', 'HMM_crowd_then_LSTM', 'bac_mace', 'bac_acc', 'bac_ibcc', 'bac_seq']#['mace', 'best', 'worst', 'majority']#'bac', 'ibcc', 'mace', 'majority'] # 'bac', 'clustering', 'ibcc', 'mace',

exp.save_results = True
exp.opt_hyper = False#True

results, preds, probs, results_nocrowd, preds_nocrowd, probs_nocrowd = exp.run_methods(annos, gt, doc_start, output_dir,
                                       text, ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val)
# for now we are just running task 1. Then we choose which version of BAC to run for task 2 with LSTM, change the
# methods and uncomment the code below.
#, gt_nocrowd, doc_start_nocrowd, text_nocrowd)
