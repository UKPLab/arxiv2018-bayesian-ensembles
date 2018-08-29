'''
Created on Oct 16, 2017

@author: Melvin Laux
'''
import os

import numpy as np
from operator import itemgetter
from itertools import groupby

from pip._vendor.distro import os_release_attr


def error_analysis(gt_path, anno_path, doc_start_path, prediction_path1, prediction_path2, output_path,
                   output_counts_path, main_method, baselines_to_skip):
    
    # load data
    gt = np.genfromtxt(gt_path, delimiter=',')

    #annos = np.genfromtxt(anno_path, delimiter=',') # can take too long to load
    annos = None

    doc_start = np.genfromtxt(doc_start_path, delimiter=',').astype(int)
    preds1 = np.genfromtxt(prediction_path1, delimiter=',', skip_header=1) # the main set of predictions
    preds2 = np.genfromtxt(prediction_path2, delimiter=',', skip_header=1) # the baselines to compare with

    # for the bio data we need to exclude the validation and unlabelled data
    ndocs = np.sum(doc_start & (gt != -1))
    ntestdocs = int(np.floor(ndocs * 0.5))
    docidxs = np.cumsum(doc_start & (gt != -1))
    ntestidxs = np.argwhere(docidxs == (ntestdocs+1))[0][0]
    gt = gt[:ntestidxs]
    doc_start = doc_start[:ntestidxs]
    doc_start = doc_start[gt != -1]
    gt = gt[gt != -1]

    # count errors of baseline predictions (methods with idx > 1 are baselines)
    num_base_errs = np.zeros_like(gt)
    for i in range(0, preds2.shape[1]):
        if i in baselines_to_skip:
            continue

        num_base_errs[np.where(preds2[:, i] != gt)] += 1

    # find all tokens correctly classified by all baselines
    base_correct = np.where(num_base_errs == 0)[0]

    # find misclassfied tokens by the main method 0
    error_idxs = np.where(preds1[:, main_method] != gt)[0]

    # find errors where the baselines were correct
    error_idxs = list(sorted(set(error_idxs).intersection(set(base_correct))))

    if annos is None:
        analysis = np.zeros((0, 4))
    else:
        analysis = np.zeros((0, 4+annos.shape[1]))

    windowsize = 10

    error_count_labels = np.array([
        'missed',
        'not a span',
        'late start',
        'early start',
        'late finish',
        'early finish',
        'fused together',
        'extra splits',
    ])
    error_counts = np.zeros(len(error_count_labels))

    analysis_header = 'tok_idx, proposed_pred, gold, doc_start'
    if annos != None:
        analysis_header += ', crowd_labels'

    for gidx, g in groupby(enumerate(error_idxs), lambda i_x:i_x[0] - i_x[1]):
        # for each error, get the predictions +/- windowsize additional tokens
        idxs = list(map(itemgetter(1), g))
        slice_ = np.s_[idxs[0] - windowsize:idxs[-1] + windowsize]
        err = np.stack((np.arange(idxs[0] - windowsize, idxs[-1] + windowsize),
                        preds1[slice_, main_method],
                        gt[slice_],
                        doc_start[slice_])).T

        # add on the original annotations
        if annos is not None:
            err = np.concatenate((err, annos[slice_, :]), axis=1)

        # add to list of error data
        analysis = np.concatenate((analysis, err), axis=0)

        # add a row of -1 to delineate the errors
        if annos is None:
            analysis = np.concatenate((analysis, -np.ones((1, 4))), axis=0)
        else:
            analysis = np.concatenate((analysis, -np.ones((1, 4 + annos.shape[1]))), axis=0)

        # categorise the error
        toks1 = preds1[idxs[0]:idxs[-1]+1, main_method]
        toks2 = gt[idxs[0]:idxs[-1]+1]

        start_displacement = 0 # positive means early, neg means late
        end_displacement = 0
        additional_splits = 0 # positive means extra, neg means missing

        p_spans = []
        span_start = -np.inf
        for i, ptok in enumerate(toks1):

            if ptok == 2:
                if span_start < -1: # else we merge the consecutive spans
                    span_start = i

            if ptok == 1 and span_start > -1:
                span_end = i
                p_spans.append((span_start, span_end))
                span_start = -1

            if ptok == 0:
                span_end = i
                g_spans.append((span_start, span_end))
                span_start = i

        # didn't finish inside the tokens
        if span_start >= -1:
            span_end = i + 1
            p_spans.append((span_start, span_end))

        g_spans = []
        span_start = -np.inf
        for i, gtok in enumerate(toks2):

            if gtok == 2:
                if span_start > -1:
                    span_end = i
                    g_spans.append((span_start, span_end))
                span_start = i

            if gtok == 1 and span_start >= -1:
                span_end = i
                g_spans.append((span_start, span_end))
                span_start = -1
            if gtok == 0 and span_start < -1:
                span_start = -1

        # didn't finish inside the tokens
        if span_start > -np.inf:
            span_end = i + 1
            g_spans.append((span_start, span_end))

        not_a_spans = 0
        missed = 0

        for i, (span_start, span_end) in enumerate(g_spans):
            # is there an exact match in p_spans?
            if (span_start, span_end) in p_spans:
                continue # not a mistake

            span_matched = False

            for pstart, pend in p_spans:
                if span_matched:
                    break

                if pstart > span_start and pstart < span_end:
                    if span_start > -1:
                        # overlaps but starts late
                        error_counts[error_count_labels == 'late start'] += 1
                        span_matched = True
                    else:
                        # overlaps but starts late
                        error_counts[error_count_labels == 'extra splits'] += 1
                        span_matched = True

                elif pstart < span_start and pend > span_start:
                    error_counts[error_count_labels == 'early start'] += 1
                    span_matched = True

                if pend < span_end and pend > span_start:
                    error_counts[error_count_labels == 'early finish'] += 1
                    span_matched = True

                elif pend > span_end and pstart < span_end:
                    error_counts[error_count_labels == 'late finish'] += 1
                    span_matched = True

                if pstart < span_end and pend > span_start:
                    span_matched = True

            if not span_matched:
                error_counts[error_count_labels == 'missed'] += 1
                missed += 1

        for i, (span_start, span_end) in enumerate(p_spans):
                # is there an exact match in p_spans?
            if (span_start, span_end) in g_spans:
                continue  # not a mistake

            span_matched = False

            for gstart, gend in g_spans:
                if gstart < span_end and gend > span_start:
                    span_matched = True

            if not span_matched:
                error_counts[error_count_labels == 'not a span'] += 1
                not_a_spans += 1

        span_count_diff = len(g_spans) - len(p_spans) + not_a_spans - missed
        error_counts[error_count_labels == 'fused together'] += (span_count_diff > 0)
        error_counts[error_count_labels == 'extra splits'] += (span_count_diff < 0)

    np.savetxt(output_path, analysis, delimiter=',', fmt='%i', header=analysis_header)
    np.savetxt(output_counts_path, error_counts[None, :], delimiter=',', fmt='%i', header=', '.join(error_count_labels))
    
if __name__ == '__main__':
    dataroot = os.path.expanduser('~/data/bayesian_annotator_combination/')

    # Analyse the errors introduced by our method that the baselines did not make.
    prior_str = 'Krusty_task2_plainpriors_proposederrors'
    error_analysis(dataroot + '/data/bio/gt.csv',
                   dataroot + '/data/bio/annos.csv',
                   dataroot + '/data/bio/doc_start.csv',
                   dataroot + '/output/bio_task2/pred_nocrowd_started-2018-08-27-13-58-22-Nseen55712.csv',
                   dataroot + '/output/bio_task2/pred_nocrowd_started-2018-08-27-13-58-22-Nseen55712.csv',
                   dataroot + '/output/bio_task2/analysis_%s' % prior_str,
                   dataroot + '/output/bio_task2/analysis_counts_%s' % prior_str,
                   0, [0])

    # Analyse the errors that our method did not make but the baselines did.
    prior_str = 'Krusty_task2_plainpriors_baselineerrors'
    error_analysis(dataroot + '/data/bio/gt.csv',
                   dataroot + '/data/bio/annos.csv',
                   dataroot + '/data/bio/doc_start.csv',
                   dataroot + '/output/bio_task2/pred_nocrowd_started-2018-08-27-13-58-22-Nseen55712.csv',
                   dataroot + '/output/bio_task2/pred_nocrowd_started-2018-08-27-13-58-22-Nseen55712.csv',
                   dataroot + '/output/bio_task2/analysis_%s' % prior_str,
                   dataroot + '/output/bio_task2/analysis_counts_%s' % prior_str,
                   1, [1])