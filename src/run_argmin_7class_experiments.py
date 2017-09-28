'''
Created on Sep 10, 2017

@author: Edwin Simpson
'''
from evaluation.experiment import Experiment
from data import load_data
import numpy as np

exp = Experiment(None, None)
exp.methods = ['bac', 'clustering', 'ibcc', 'mace' , 'majority'] # 'bac', 'clustering', 'ibcc', 'mace', 

gt, annos, doc_start = load_data.load_argmin_7class_data()

exp.save_results = True
exp.num_classes = 7

exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, annos.shape[1])) +  1.0 * np.eye(exp.num_classes)[:,:,None,None]

results, preds = exp.run_methods(annos, gt, doc_start, -666, '../data/argmin7/annos.csv')

np.savetxt('../output/argmin7/result_full', results, fmt='%s', delimiter=',')
np.savetxt('../output/argmin7/pred_full', preds, fmt='%s', delimiter=',')

#results.dump('../output/argmin/results')
#preds.dump('../output/argmin/preds')

if __name__ == '__main__':
    pass
