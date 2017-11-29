'''
Created on Oct 25, 2016

@author: Melvin Laux
'''

from data.load_data import load_crowdsourcing_data, load_argmin_data
from algorithm import bac
from baselines import majority_voting
import numpy as np

# load data
annos, doc_start = load_crowdsourcing_data()

# run majority voting
base = majority_voting.MajorityVoting(annos, 3)
maj, votes = base.vote()

# run BAC 
bac_ = bac.BAC(L=3, K=annos.shape[1])
probs, agg = bac_.run(annos, doc_start)

np.savetxt('./data/crowdsourcing/gen/probs2.csv', probs, delimiter=',')
np.savetxt('./data/crowdsourcing/gen/agg2.csv', agg, delimiter=',')

np.savetxt('./data/crowdsourcing/gen/majority2.csv', maj , delimiter=',')
np.savetxt('./data/crowdsourcing/gen/votes2.csv', votes, delimiter=',')

# load data
# probs = np.genfromtxt('./data/crowdsourcing/gen/probs.csv', delimiter=',')
# agg = np.genfromtxt('./data/crowdsourcing/gen/agg.csv', delimiter=',')

# build gold standard
gold = -np.ones_like(agg)

for i in xrange(agg.shape[0]):
    if ((agg[i] == int(maj[i])) and (i, probs[int(agg[i])] > 0.9)):
        gold[i] = agg[i]
        
np.savetxt('./data/crowdsourcing/gen/gold2.csv', gold, fmt='%s', delimiter=',')




if __name__ == '__main__':
    pass
