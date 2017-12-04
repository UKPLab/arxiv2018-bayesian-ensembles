'''
Created on Aug 8, 2017

@author: Melvin Laux
'''
from evaluation.experiment import Experiment
from data import load_data
import numpy as np

exp = Experiment(None, None)
exp.methods = ['bac', 'ibcc', 'mace', 'majority'] # 'bac', 'clustering', 'ibcc', 'mace', 

gt, annos, doc_start = load_data.load_argmin_data()

exp.save_results = True
exp.num_classes = 3

# priorstr = 'prior_1' # ones along diagonals, alpha0 excludes transitions from outside (1 or -1) to inside (0)
# exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
#                           annos.shape[1])) +  1.0 * np.eye(exp.num_classes)[:,:,None,None]

# TODO: why is the result different from the prior_2 result if we haven't changed anything? Something must have changed.
priorstr = 'prior_2_try2' # fives along diagonals, alpha0 excludes transitions from outside (1 or -1) to inside (0)
exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
                          annos.shape[1])) +  5.0 * np.eye(exp.num_classes)[:,:,None,None]

# priorstr = 'prior_3' # We address the imbalance between rows with exclusions and rows without. When there is an exclusion,
# # weights on the correct answer are 0.667 instead of 0.5. Weights on the incorrect answer are 0.333 instead of 0.25, so
# # the correct answer benefits more. Solution: put the pseudo-counts from the excluded transition onto the next-best answer,
# # e.g. the best answer given that the previous one had missed the start of the annotation.
# # BAC has now been changed to make this work: 
# exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
#                            annos.shape[1])) +  1.0 * np.eye(exp.num_classes)[:,:,None,None]

# priorstr = 'prior_4' 
# exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
#                            annos.shape[1])) +  5.0 * np.eye(exp.num_classes)[:,:,None,None]

# priorstr = 'prior_5' 
# exp.bac_alpha0 = 0.1 * np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
#                            annos.shape[1])) +  0.1 * np.eye(exp.num_classes)[:,:,None,None]

results, preds, _ = exp.run_methods(annos, gt, doc_start, -666, './data/argmin/annos.csv')


np.savetxt('./output/argmin/result_err_%s' % priorstr, results, fmt='%s', delimiter=',')
np.savetxt('./output/argmin/pred_err_%s' % priorstr, preds, fmt='%s', delimiter=',')

#results.dump('./output/argmin/results')
#preds.dump('./output/argmin/preds')

if __name__ == '__main__':
    pass
