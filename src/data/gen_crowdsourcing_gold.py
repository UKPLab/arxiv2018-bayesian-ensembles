'''
Created on Oct 25, 2016

@author: Melvin Laux
'''

from data.load_data import load_crowdsourcing_data
import algorithm.bsc
import baselines.majority_voting
import numpy as np
import data.data_utils as data_utils

# load data
annos, doc_start = load_crowdsourcing_data()
N = float(annos.shape[0])
K = annos.shape[1]
L = 3

# run majority voting
base = majority_voting.MajorityVoting(annos, 3)
maj, votes = base.vote()
maj = data_utils.postprocess(maj, doc_start)

# run BAC 
bac_ = bac.BAC(L=L, K=K, nu0=np.ones((L+1, L)) * 100, alpha0=100.0 * (np.ones((L, L, L+1, K)) + 1.0 * np.eye(3)[:, :, None, None]))
probs, agg = bac_.run_synth(annos, doc_start)

np.savetxt('../../data/bayesian_annotator_combination/data/crowdsourcing/gen/probs2.csv', probs, delimiter=',')
np.savetxt('../../data/bayesian_annotator_combination/data/crowdsourcing/gen/agg2.csv', agg, delimiter=',')

np.savetxt('../../data/bayesian_annotator_combination/data/crowdsourcing/gen/majority2.csv', maj , delimiter=',')
np.savetxt('../../data/bayesian_annotator_combination/data/crowdsourcing/gen/votes2.csv', votes, delimiter=',')

# load data
# probs = np.genfromtxt('../../data/bayesian_annotator_combination/data/crowdsourcing/gen/probs.csv', delimiter=',')
# agg = np.genfromtxt('../../data/bayesian_annotator_combination/data/crowdsourcing/gen/agg.csv', delimiter=',')

# build gold standard
gold = -np.ones_like(agg)

for i in range(agg.shape[0]):
    if ((agg[i] == int(maj[i])) and (i, probs[int(agg[i])] > 0.5)):
        gold[i] = agg[i]
        
print("Generated the gold data for testing on the crowdsourcing dataset. There are %i gold-labelled and %i unconfirmed data points." % (np.sum(gold!=-1), np.sum(gold==-1)))    
    
np.savetxt('../../data/bayesian_annotator_combination/data/crowdsourcing/gen/gold2.csv', gold, fmt='%s', delimiter=',')

if __name__ == '__main__':
    pass
