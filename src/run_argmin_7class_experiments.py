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

results, preds = exp.run_methods(annos, gt, doc_start, -666, '../data/argmin7/annos.csv')

np.savetxt('../output/argmin7/result_full', results, fmt='%s', delimiter=',')
np.savetxt('../output/argmin7/pred_full', preds, fmt='%s', delimiter=',')

#results.dump('../output/argmin/results')
#preds.dump('../output/argmin/preds')

if __name__ == '__main__':
    pass
