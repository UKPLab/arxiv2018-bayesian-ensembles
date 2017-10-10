'''
Created on Sep 26, 2017

@author: Melvin Laux
'''
from evaluation.experiment import Experiment
from data import load_data
import numpy as np
import os

output_dir = '../output/argmin7/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

exp = Experiment(None, None)
exp.methods = ['bac'] # 'bac', 'clustering', 'ibcc', 'mace', 

gt, annos, doc_start = load_data.load_argmin_7class_data()

exp.save_results = True
exp.num_classes = 7

exp.exclusions = {0:[3,4,5,6],2:[3,4,5,6],3:[0,2,5,6],4:[0,2,5,6],5:[0,2,3,4],6:[0,2,3,4]}


exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, annos.shape[1])) +  50000.0 * np.eye(exp.num_classes)[:,:,None,None]

results, preds, _ = exp.run_methods(annos, gt, doc_start, -666, '../data/argmin7/annos.csv')

np.savetxt(output_dir + 'result_7class_prior_50000', results, fmt='%s', delimiter=',')
np.savetxt(output_dir + 'pred_7class_prior_50000', preds, fmt='%s', delimiter=',')


#results.dump('../output/argmin/results')
#preds.dump('../output/argmin/preds')

if __name__ == '__main__':
    pass
