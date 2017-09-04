'''
Created on Aug 8, 2017

@author: Melvin Laux
'''
from evaluation.experiment import Experiment
from data import load_data


exp = Experiment(None, None)
exp.methods = ['ibcc', 'mace', 'majority']

gt, annos, doc_start = load_data.load_argmin_data()

exp.save_results = True
exp.num_classes = 3

results  = exp.run_methods(annos, gt, doc_start, -666, '../data/argmin/annos.csv')

results.dump('../output/argmin/')

if __name__ == '__main__':
    pass