'''
Created on Sep 4, 2017

@author: Melvin Laux
'''
import numpy as np
import itertools
import os
from random import shuffle
from evaluation.experiment import Experiment

# load data
gold = np.genfromtxt('../data/crowdsourcing/gen/gold.csv', delimiter=',')
annos = np.genfromtxt('../data/crowdsourcing/gen/annos.csv', delimiter=',')
doc_start = np.genfromtxt('../data/crowdsourcing/gen/doc_start.csv', delimiter=',')

# setup
num_tokens = annos.shape[0]
num_annos = annos.shape[1]
l = range(num_annos)

exp = Experiment(None, None)
exp.methods = ['bac', 'ibcc', 'mace', 'majority']
exp.num_classes = 3

for k in xrange(3, 25, 3):
    shuffle(l)
    it = itertools.combinations(l, k)
    for run in xrange(10):
        output_dir = '../output/crowdsourcing/k' + str(k) + '/run' + str(run) + '/'
        subannos = annos[:, it.next()]
        subannos_str = subannos.astype(str)
        subannos_str[subannos_str == '-1.0'] = ''

        if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        np.savetxt(output_dir + 'annos.csv', subannos_str, fmt='%s', delimiter=',')
        results, preds = exp.run_methods(subannos, gold, doc_start[:, None], -666, output_dir + 'annos.csv')
        np.savetxt(output_dir + 'results.csv', results, fmt='%s', delimiter=',')
        np.savetxt(output_dir + 'preds.csv', preds, fmt='%s', delimiter=',')

if __name__ == '__main__':
    pass
