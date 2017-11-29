'''
Created on Sep 4, 2017

@author: Melvin Laux
'''
import numpy as np
import os
from evaluation.experiment import Experiment

# load data
gold = np.genfromtxt('./data/crowdsourcing/gen/gold2.csv', delimiter=',')
annos = np.genfromtxt('./data/crowdsourcing/gen/annos.csv', delimiter=',')
doc_start = np.genfromtxt('./data/crowdsourcing/gen/doc_start.csv', delimiter=',')

# setup
num_tokens = annos.shape[0]
num_annos = annos.shape[1]

exp = Experiment(None, None)
exp.methods = ['bac', 'clustering', 'HMM_crowd', 'ibcc', 'mace', 'majority']
exp.num_classes = 3

annosA = annos[:, :24]
annosB = annos[:, 24:]

# create list of doc start idxs

doc_start_idxs = np.where(doc_start == 1)[0]
np.append(doc_start_idxs, annosA.shape[0])

dataA = -np.ones_like(annosA)
dataB = -np.ones_like(annosB)

for k in xrange(1, 9):    
    
    for i in xrange(len(doc_start_idxs) - 1):
        
        col = np.where(np.cumsum(annosA[doc_start_idxs[i], :] != -1) == k)[0][0]
        dataA[doc_start_idxs[i]:doc_start_idxs[i + 1] - 1, col] = annosA[doc_start_idxs[i]:doc_start_idxs[i + 1] - 1, col]
    
        col = np.where(np.cumsum(annosB[doc_start_idxs[i], :] != -1) == k)[0][0]
        dataB[doc_start_idxs[i]:doc_start_idxs[i + 1] - 1, col] = annosB[doc_start_idxs[i]:doc_start_idxs[i + 1] - 1, col]
    
    output_dir = './output/crowdsourcing/k' + str(k) + '/run0/'
    subannos = dataA
    subannos_str = subannos.astype(str)
    subannos_str[subannos_str == '-1.0'] = ''

    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    np.savetxt(output_dir + 'annos2.csv', subannos_str, fmt='%s', delimiter=',')
    results, preds, probs = exp.run_methods(subannos, gold, doc_start[:, None], -666, output_dir + 'annos2.csv')
    np.savetxt(output_dir + 'results2.csv', results, fmt='%s', delimiter=',')
    np.savetxt(output_dir + 'preds2.csv', preds, fmt='%s', delimiter=',')
    
    output_dir = './output/crowdsourcing/k' + str(k) + '/run1/'
    subannos = dataB
    subannos_str = subannos.astype(str)
    subannos_str[subannos_str == '-1.0'] = ''

    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    np.savetxt(output_dir + 'annos2.csv', subannos_str, fmt='%s', delimiter=',')
    results, preds, probs = exp.run_methods(subannos, gold, doc_start[:, None], -666, output_dir + 'annos2.csv')
    np.savetxt(output_dir + 'results2.csv', results, fmt='%s', delimiter=',')
    np.savetxt(output_dir + 'preds2.csv', preds, fmt='%s', delimiter=',')

if __name__ == '__main__':
    pass
