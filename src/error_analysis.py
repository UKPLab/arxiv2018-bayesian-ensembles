'''
Created on Oct 16, 2017

@author: Melvin Laux
'''

import numpy as np
from operator import itemgetter
from itertools import groupby

def error_analysis(gt_path, anno_path, doc_start_path, prediction_path, output_path):
    
    # load data
    gt = np.genfromtxt(gt_path, delimiter=',')
    annos = np.genfromtxt(anno_path, delimiter=',')
    doc_start = np.genfromtxt(doc_start_path, delimiter=',')
    preds = np.genfromtxt(prediction_path, delimiter=',')

    # count errors of baseline predictions 
    num_base_errs = np.zeros_like(gt)
    for i in xrange(1, preds.shape[1]):
        num_base_errs[np.where(preds[:,i]!=gt)] += 1

    # find all tokens correctly classified by all baselines
    base_correct = np.where(num_base_errs==0)[0]

    # find misclassfied tokens
    error_idxs = np.where(preds[:,0]!=gt)[0]
    error_idxs = list(sorted(set(error_idxs).intersection(set(base_correct))))

    analysis = np.zeros((0,4+annos.shape[1]))

    for _, g in groupby(enumerate(error_idxs), lambda (i,x):i-x):
        idxs = map(itemgetter(1), g)
        slice_ = np.s_[idxs[0]-10:idxs[-1]+10]
        err = np.stack((np.arange(idxs[0]-10,idxs[-1]+10),preds[slice_,0],gt[slice_],doc_start[slice_])).T
        err = np.concatenate((err,annos[slice_,:]), axis=1)
        analysis = np.concatenate((analysis,err),axis=0)
        analysis = np.concatenate((analysis,-np.ones((1,4 + annos.shape[1]))),axis=0)

    np.savetxt(output_path, analysis, delimiter=',', fmt='%i')
    
if __name__ == '__main__':
    prior_str = 'prior_3'
    error_analysis('./data/argmin/gt.csv', './data/argmin/annos.csv', './data/argmin/doc_start.csv', 
                   './output/argmin/pred_err_%s' % prior_str, './output/argmin/analysis_%s' % prior_str)
