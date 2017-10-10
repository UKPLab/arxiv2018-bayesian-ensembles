'''
Created on Aug 8, 2017

@author: Melvin Laux
'''
from evaluation.experiment import Experiment
from data import load_data
import numpy as np

exp = Experiment(None, None)
exp.methods = ['majority'] # 'bac', 'clustering', 'ibcc', 'mace', 

gt, annos, doc_start = load_data.load_argmin_data()

exp.save_results = True
exp.num_classes = 3

exp.bac_alpha0 = np.ones((exp.num_classes, exp.num_classes, exp.num_classes+1, 
                          annos.shape[1])) +  1.0 * np.eye(exp.num_classes)[:,:,None,None]                   

results, preds, _ = exp.run_methods(annos, gt, doc_start, -666, '../data/argmin/annos.csv')

np.savetxt('../output/argmin/result_maj', results, fmt='%s', delimiter=',')
np.savetxt('../output/argmin/pred_maj', preds, fmt='%s', delimiter=',')

#results.dump('../output/argmin/results')
#preds.dump('../output/argmin/preds')

if __name__ == '__main__':
    pass
