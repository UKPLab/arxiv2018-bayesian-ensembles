'''
Created on Oct 25, 2016

@author: Melvin Laux
'''

from data.load_data import load_crowdsourcing_data
from algorithm import bac
import numpy as np

annos, doc_start = load_crowdsourcing_data()

alg = bac.BAC(L=3, K=annos.shape[1])
probs = alg.run(annos, doc_start)

np.savetxt('../data/crowdsourcing/probs.csv', probs, fmt='%s', delimiter=',')


if __name__ == '__main__':
    pass
