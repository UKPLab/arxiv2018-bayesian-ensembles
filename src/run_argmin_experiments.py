'''
Created on Aug 8, 2017

@author: Melvin Laux
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

gt, annos, doc_start = load_data.load_argmin_data()

exp = Experiment(None, 3, annos.shape[1], None)
exp.methods = ['bac_acc']#['mace', 'best', 'worst', 'majority']#'bac', 'ibcc', 'mace', 'majority'] # 'bac', 'clustering', 'ibcc', 'mace',

exp.save_results = True
exp.opt_hyper = True

# priorstr = 'prior_1_try2' # ones along diagonals, alpha0 excludes transitions from outside (1 or -1) to inside (0)
# exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
#                           annos.shape[1])) +  1.0 * np.eye(exp.num_classes)[:,:,None,None]
#
# why is the result different from the prior_2 result if we haven't changed anything? Something must have changed.
# priorstr = 'prior_2_try2' # fives along diagonals, alpha0 excludes transitions from outside (1 or -1) to inside (0)
# exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
#                           annos.shape[1])) +  5.0 * np.eye(exp.num_classes)[:,:,None,None]
#
# priorstr = 'prior_3_try2' # We address the imbalance between rows with exclusions and rows without. When there is an exclusion,
# # weights on the correct answer are 0.667 instead of 0.5. Weights on the incorrect answer are 0.333 instead of 0.25, so
# # the correct answer benefits more. Solution: put the pseudo-counts from the excluded transition onto the next-best answer,
# # e.g. the best answer given that the previous one had missed the start of the annotation.
# # BAC has now been changed to make this work: 
# exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
#                             annos.shape[1])) +  1.0 * np.eye(exp.num_classes)[:,:,None,None]
#
# priorstr = 'prior_4_try2'
# exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1,
#                            annos.shape[1])) +  5.0 * np.eye(exp.num_classes)[:,:,None,None]
#
# priorstr = 'prior_5_try2'
# exp.bac_alpha0 = 0.1 * np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
#                            annos.shape[1])) +  0.1 * np.eye(exp.num_classes)[:,:,None,None]

output_dir = '../../data/bayesian_annotator_combination/output/argmin/'

results, preds, _ = exp.run_methods(annos, gt, doc_start, output_dir)

if __name__ == '__main__':
    pass
