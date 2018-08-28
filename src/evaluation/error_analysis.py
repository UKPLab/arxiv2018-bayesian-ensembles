'''
Created on Oct 16, 2017

@author: Melvin Laux
'''
import os

import numpy as np
from operator import itemgetter
from itertools import groupby

from pip._vendor.distro import os_release_attr


def error_analysis(gt_path, anno_path, doc_start_path, prediction_path, output_path):
    
    # load data
    gt = np.genfromtxt(gt_path, delimiter=',')
    annos = np.genfromtxt(anno_path, delimiter=',')
    doc_start = np.genfromtxt(doc_start_path, delimiter=',')
    preds = np.genfromtxt(prediction_path, delimiter=',')

    main_method = 0

    # count errors of baseline predictions (methods with idx > 1 are baselines)
    num_base_errs = np.zeros_like(gt)
    for i in range(0, preds.shape[1]):
        if i == main_method:
            continue

        num_base_errs[np.where(preds[:, i] != gt)] += 1

    # find all tokens correctly classified by all baselines
    base_correct = np.where(num_base_errs == 0)[0]

    # find misclassfied tokens by the main method 0
    error_idxs = np.where(preds[:, main_method] != gt)[0]

    # find errors where the baselines were correct
    error_idxs = list(sorted(set(error_idxs).intersection(set(base_correct))))

    analysis = np.zeros((0, 4+annos.shape[1]))

    windowsize = 10

    for _, g in groupby(enumerate(error_idxs), lambda i_x:i_x[0]-i_x[1]):
        # for each error, get the predictions +/- windowsize additional tokens
        idxs = list(map(itemgetter(1), g))
        slice_ = np.s_[idxs[0]-windowsize:idxs[-1]+windowsize]
        err = np.stack((np.arange(idxs[0]-windowsize, idxs[-1]+windowsize),preds[slice_,0],gt[slice_],doc_start[slice_])).T

        # add on the original annotations
        err = np.concatenate((err, annos[slice_,:]), axis=1)

        # add to list of error data
        analysis = np.concatenate((analysis,err),axis=0)

        # add a row of -1 to delineate the errors
        analysis = np.concatenate((analysis,-np.ones((1, 4 + annos.shape[1]))),axis=0)

    np.savetxt(output_path, analysis, delimiter=',', fmt='%i')
    
if __name__ == '__main__':
    prior_str = 'Krusty_task2_plain_priors'

    dataroot = os.path.expanduser('~/data/bayesian_annotator_combination/')

    error_analysis(dataroot + '/data/bio/gt.csv',
                   dataroot + '/data/bio/annos.csv',
                   dataroot + '/data/bio/doc_start.csv',
                   dataroot + '/output/bio_task2/pred_nocrowd_started-2018-08-27-13-58-22-Nseen55712.csv',
                   dataroot + '/output/bio_task2/analysis_%s' % prior_str)
