'''
Created on Oct 16, 2017

@author: Melvin Laux
'''

import numpy as np
from data import load_data
from operator import itemgetter
from itertools import groupby

#gt, annos, doc_start = load_data.load_argmin_data()
gt = np.genfromtxt('../data/argmin/gt.csv', delimiter=',')
annos = np.genfromtxt('../data/argmin/annos.csv', delimiter=',')
doc_start = np.genfromtxt('../data/argmin/doc_start.csv', delimiter=',')

results = np.genfromtxt('../output/argmin/result_err_prior_1', delimiter=',')
preds = np.genfromtxt('../output/argmin/pred_err_prior_1', delimiter=',')

num_base_errs = np.zeros_like(gt)
num_base_errs[np.where(preds[:,1]!=gt)] += 1
num_base_errs[np.where(preds[:,2]!=gt)] += 1
num_base_errs[np.where(preds[:,3]!=gt)] += 1

base_correct = np.where(num_base_errs==0)[0]

error_idxs = np.where(preds[:,0]!=gt)[0]
error_idxs = list(sorted(set(error_idxs).intersection(set(base_correct))))

analysis = np.zeros((0,4+annos.shape[1]))

for k, g in groupby(enumerate(error_idxs), lambda (i,x):i-x):
    idxs = map(itemgetter(1), g)
    slice_ = np.s_[idxs[0]-10:idxs[-1]+10]
    err = np.stack((np.arange(idxs[0]-10,idxs[-1]+10),preds[slice_,0],gt[slice_],doc_start[slice_])).T
    err = np.concatenate((err,annos[slice_,:]), axis=1)
    analysis = np.concatenate((analysis,err),axis=0)
    analysis = np.concatenate((analysis,-np.ones((1,4 + annos.shape[1]))),axis=0)

np.savetxt('../output/argmin/analysis_prior_1', analysis, delimiter=',', fmt='%i')

if __name__ == '__main__':
    pass