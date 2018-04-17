'''
Created on Sep 4, 2017

@author: Melvin Laux
'''
import numpy as np
import os
from evaluation.experiment import Experiment

L = 3

# TODO: optimising hyperparameters: the best values seem to be around 16 for the alphafactor. When you use this value, 
# the sum of all pseudocounts is comparable to the number of data points. Is the balance somehow important to prevent
# overfitting while still permitting learning?
# TODO: LB just favours small alpha0 values. Is there a principled choice of hyper-prior that would balance this? E.g.
# assume that the workers vary between 50% and 100% correct so that the mean is 75% correct.  The larger
# alpha factor is needed because the shape of the PDF changes so that (a) accuracy < 50% is very unlikely and accuracy
# ? 90% is very unlikely (if the factor is 1, the most likely accuracy is 1) ==> more appropriate mode and spread.
# Check this out by looking at the CDFs -- with alpha factor 1, the probability of a worker being worse than random is
# approximately 0.25 for various different nos. classes. With alpha factor 16, the probability is < 0.01. For a spammer
# who always clicks the same answer, we require many more than alpha factor data points to learn the spamming behaviour.
# Do these priors work in all situations? 
# - if the size of the dataset is larger, do we need to increase the factor to compensate? Probably unnecessary if we 
# use Gibbs' sampling, but this may offset the possibility of falling into a bad unimodal solution using VB. 
# - setting an overly large alpha factor may be less of a problem than setting a small one because decisions still go 
# in the right direction (albeit miscalibrated probabilities) when alpha is larger than optimal, but could end up 
# flipped if alpha is too small. This is born out in reasonable accuracy with larger alpha factor while log loss 
# increases.
# A clue: cross entropy error is a good indicator of the overall best performance. Does it also correlate with predicted
# label entropy, i.e. avoiding strong classifications?

# load data
gold = np.genfromtxt('./data/crowdsourcing/gen/gold2.csv', delimiter=',')
annos = np.genfromtxt('./data/crowdsourcing/gen/annos.csv', delimiter=',')
doc_start = np.genfromtxt('./data/crowdsourcing/gen/doc_start.csv', delimiter=',')

# setup
num_tokens = annos.shape[0]
num_annos = annos.shape[1]

exp = Experiment(None, None)
exp.methods = ['bac']
exp.num_classes = 3

annosA = annos[:, :24]
annosB = annos[:, 24:]

# create list of doc start idxs

doc_start_idxs = np.where(doc_start == 1)[0]
doc_end_idxs = np.append(doc_start_idxs[1:], len(doc_start))
np.append(doc_start_idxs, annosA.shape[0])

dataA = -np.ones_like(annosA)
dataB = -np.ones_like(annosB)

# test different hyperparameter settings
alpha0_factors = np.arange(1, 20) ** 2 # 100.0 --> the working value

alpha0_diags = 1.0
nu0_factor = 100.0    

kmax = 3 # number of annotators
for k in xrange(1, kmax):    
    for i in xrange(len(doc_start_idxs)):
    
        col = np.where(np.cumsum(annosA[doc_start_idxs[i], :] != -1) == k)[0][0]
        dataA[doc_start_idxs[i]:doc_end_idxs[i], col] = annosA[doc_start_idxs[i]:doc_end_idxs[i], col]

        col = np.where(np.cumsum(annosB[doc_start_idxs[i], :] != -1) == k)[0][0]
        dataB[doc_start_idxs[i]:doc_end_idxs[i], col] = annosB[doc_start_idxs[i]:doc_end_idxs[i], col]

output_dir = './output/hyperparameters_crowdsourcing/k' + str(k) + '/run0/'
subannos = dataA
subannos_str = subannos.astype(str)
subannos_str[subannos_str == '-1.0'] = ''

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

np.savetxt(output_dir + 'annos2.csv', subannos_str, fmt='%s', delimiter=',')

for alpha0_factor in alpha0_factors:
    exp.bac_alpha0 = alpha0_factor * (np.ones((3, 3, 4, subannos.shape[1])) + alpha0_diags * np.eye(3)[:, :, None, None])
    exp.bac_nu0 = np.ones((L + 1, L)) * nu0_factor    
    results, preds, probs, model = exp.run_methods(subannos, gold, doc_start[:, None], -666, output_dir + 'annos2.csv', return_model=True)

    output_dir = './output/hyperparameters_crowdsourcing/alphafactor' + str(alpha0_factor) + '/run0/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = np.append(results, [[model.lowerbound()]], axis=0)
    np.savetxt(output_dir + 'results2.csv', results, fmt='%s', delimiter=',')
    np.savetxt(output_dir + 'preds2.csv', preds, fmt='%s', delimiter=',')
    np.savetxt(output_dir + 'probs2.csv', probs.reshape(probs.shape[0], probs.shape[1] * probs.shape[2]), fmt='%s', delimiter=',')

output_dir = './output/hyperparameters_crowdsourcing/k' + str(k) + '/run1/'
subannos = dataB
subannos_str = subannos.astype(str)
subannos_str[subannos_str == '-1.0'] = ''

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

np.savetxt(output_dir + 'annos2.csv', subannos_str, fmt='%s', delimiter=',')

for alpha0_factor in alpha0_factors:
    exp.bac_alpha0 = alpha0_factor * (np.ones((3, 3, 4, subannos.shape[1])) + alpha0_diags * np.eye(3)[:, :, None, None])
    exp.bac_nu0 = np.ones((L + 1, L)) * nu0_factor
    results, preds, probs, model = exp.run_methods(subannos, gold, doc_start[:, None], -666, output_dir + 'annos2.csv', return_model=True)
    
    output_dir = './output/hyperparameters_crowdsourcing/alphafactor' + str(alpha0_factor) + '/run1/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    results = np.append(results, [[model.lowerbound()]], axis=0)
    np.savetxt(output_dir + 'results2.csv', results, fmt='%s', delimiter=',')
    np.savetxt(output_dir + 'preds2.csv', preds, fmt='%s', delimiter=',')
    np.savetxt(output_dir + 'probs2.csv', probs.reshape(probs.shape[0], probs.shape[1] * probs.shape[2]), fmt='%s', delimiter=',')

if __name__ == '__main__':
    pass
