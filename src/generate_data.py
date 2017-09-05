'''
Created on Oct 25, 2016

@author: Melvin Laux
'''

from data.load_data import load_crowdsourcing_data
from algorithm import bac
from baselines import majority_voting
import numpy as np

# load data
annos, doc_start = load_crowdsourcing_data()

# run majority voting
base = majority_voting.MajorityVoting(annos, 3)
maj, votes = base.vote()

np.savetxt('../data/crowdsourcing/gen/majority.csv', maj , delimiter=',')
np.savetxt('../data/crowdsourcing/gen/votes.csv', votes, delimiter=',')


# load data
probs = np.genfromtxt('../data/crowdsourcing/gen/probs.csv', delimiter=',')
agg = np.genfromtxt('../data/crowdsourcing/gen/agg.csv', delimiter=',')

# build gold standard
gold = -np.ones_like(agg)

for i in xrange(agg.shape[0]):
    if ((agg[i] == int(maj[i])) and (i, probs[int(agg[i])] > 0.9)):
        gold[i] = agg[i]
        
np.savetxt('../data/crowdsourcing/gen/gold.csv', gold, fmt='%s', delimiter=',')




if __name__ == '__main__':
    pass
