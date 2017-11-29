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

# TODO: address lower bound bug by holding each variable fixed and testing in turn. Then rerun prior_1 and prior_2. 

# priorstr = 'prior_1' # ones along diagonals, alpha0 excludes transitions from outside (1 or -1) to inside (0)
# exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
#                           annos.shape[1])) +  1.0 * np.eye(exp.num_classes)[:,:,None,None]
# exp.exclusions = {}


priorstr = 'prior_2' # fives along diagonals, alpha0 excludes transitions from outside (1 or -1) to inside (0)
exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
                          annos.shape[1])) +  5.0 * np.eye(exp.num_classes)[:,:,None,None]
exp.exclusions = {}

# priorstr = 'prior_3' # We address the imblance between rows with exclusions and rows without. When there is an exclusion,
# weights on the correct answer are 0.667 instead of 0.5. Weights on the incorrect answer are 0.333 instead of 0.25, so
# the correct answer benefits more; should put more weight onto chains of O labels being correct? 
# TODO: How to address this below? 
# exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
#                           annos.shape[1])) +  1.0 * np.eye(exp.num_classes)[:,:,None,None]
# exp.exclusions = {}

results, preds, _ = exp.run_methods(annos, gt, doc_start, -666, './data/argmin/annos.csv')


np.savetxt('./output/argmin/result_err_%s' % priorstr, results, fmt='%s', delimiter=',')
np.savetxt('./output/argmin/pred_err_%s' % priorstr, preds, fmt='%s', delimiter=',')

#results.dump('./output/argmin/results')
#preds.dump('./output/argmin/preds')

if __name__ == '__main__':
    pass
