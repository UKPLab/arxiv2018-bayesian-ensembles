'''
Created on Oct 16, 2017

@author: Melvin Laux
'''
import os

import numpy as np
from operator import itemgetter
from itertools import groupby

from pip._vendor.distro import os_release_attr


def error_analysis(gt_path, anno_path, doc_start_path, prediction_path1, output_path,
                   output_counts_path, main_method, baselines_to_skip=None, compare_to_gold_only=True,
                   prediction_path2=None, remove_val=False):
    
    # load data
    gt = np.genfromtxt(gt_path, delimiter=',')
    if gt.ndim == 2 and gt.shape[1] == 2:
        gt = gt[:, 1]
        gt = gt[1:]
    #annos = np.genfromtxt(anno_path, delimiter=',') # can take too long to load
    annos = None

    doc_start = np.genfromtxt(doc_start_path, delimiter=',').astype(int)

    if doc_start.ndim == 2 and doc_start.shape[1] == 2:
        doc_start = doc_start[:, 1]
        doc_start = doc_start[1:]

    preds1 = np.genfromtxt(prediction_path1, delimiter=',', skip_header=1) # the main set of predictions
    if preds1.ndim > 1:
        preds1 = preds1[:, main_method]
    if len(preds1) > len(gt):
        print('Using only the first predictions to match length of gold standard labels...')
        preds1 = preds1[:len(gt)]

    if not compare_to_gold_only:
        preds2 = np.genfromtxt(prediction_path2, delimiter=',', skip_header=1) # the baselines to compare with

    # for the bio data we need to exclude the validation and unlabelled data
    if remove_val:
        ndocs = np.sum(doc_start & (gt != -1))
        ntestdocs = int(np.floor(ndocs * 0.5))
        docidxs = np.cumsum(doc_start & (gt != -1))
        ntestidxs = np.argwhere(docidxs == (ntestdocs+1))[0][0]

        gt = gt[:ntestidxs]

        doc_start = doc_start[:ntestidxs]
        doc_start = doc_start[gt != -1]

        preds1 = preds1[:ntestidxs]
        preds1 = preds1[gt != -1]

        gt = gt[gt != -1]


    print('Are the datasets the same shape?')
    print(preds1.shape)
    print(gt.shape)

    # find misclassfied tokens by the main method 0
    error_idxs = np.where(preds1 != gt)[0]

    # count errors of baseline predictions (methods with idx > 1 are baselines)
    if not compare_to_gold_only:
        num_base_errs = np.zeros_like(gt)
        for i in range(0, preds2.shape[1]):
            if i in baselines_to_skip:
                continue

            num_base_errs[np.where(preds2[:, i] != gt)] += 1

        # find all tokens correctly classified by all baselines
        base_correct = np.where(num_base_errs == 0)[0]

        # find errors where the baselines were correct
        error_idxs = list(sorted(set(error_idxs).intersection(set(base_correct))))

    if annos is None:
        analysis = np.zeros((0, 4))
    else:
        analysis = np.zeros((0, 4+annos.shape[1]))

    windowsize = 10

    error_count_labels = np.array([
        'exact match',
        'wrong type',
        'partial match',
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

    g_spans = []
    g_spans_with_error = []
    g_span_partial_match = []

    span_start = -1

    has_error = False
    partial_match = False

    for i, gtok in enumerate(gt):

        if gtok in [2, 4, 6, 8]:
            # this is the end of any current spans
            if span_start > -1:
                span_end = i
                g_spans.append((span_start, span_end))
                g_spans_with_error.append(has_error)
                g_span_partial_match.append(
                    partial_match and has_error)  # include spans with some matching tokens but some errors
            # this is the beginning of a new span
            span_start = i
            if i in error_idxs: # record if the prediction contains an error
                has_error = True
                partial_match = False
            else:
                partial_match = True
                has_error = False

        elif gtok == 1 and span_start > -1:
            # an O token -- end any current spans
            span_end = i
            g_spans.append((span_start, span_end))
            g_spans_with_error.append(has_error)
            g_span_partial_match.append(
                partial_match and has_error) #include spans with some matching tokens but some errors
            span_start = -1

        elif span_start > -1 and i in error_idxs:
            # we have a gold I token and an error in the prediction
            has_error = True

        elif span_start > -1:
            partial_match = True # this I token matches, so there is a partial match at least

    # didn't finish inside the tokens
    if span_start > -1:
        span_end = i + 1
        g_spans.append((span_start, span_end))
        g_spans_with_error.append(has_error)
        g_span_partial_match.append(
            partial_match and has_error)  # include spans with some matching tokens but some errors

    error_counts[error_count_labels == 'exact match'] = len(g_spans) - np.sum(g_spans_with_error)
    error_counts[error_count_labels == 'partial match'] = np.sum(g_span_partial_match)
    print('No. gold spans with no errors = %i' % error_counts[error_count_labels == 'exact match'])
    print('Total number of spans in the gold data = %i' % len(g_spans))

    analysis_header = 'tok_idx, proposed_pred, gold, doc_start'
    if annos != None:
        analysis_header += ', crowd_labels'

    for gidx, g in groupby(enumerate(error_idxs), lambda i_x:i_x[0] - i_x[1]):
        # for each error, get the predictions +/- windowsize additional tokens
        idxs = list(map(itemgetter(1), g))
        start = idxs[0] - windowsize
        if start < 0:
            start = 0

        fin = idxs[-1] + windowsize
        if fin > len(gt):
            fin = len(gt)

        slice_ = np.s_[start:fin]
        err = np.stack((np.arange(start, fin),
                        preds1[slice_],
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
        toks1 = preds1[idxs[0]:idxs[-1]+1].copy()
        toks2 = gt[idxs[0]:idxs[-1]+1].copy()

        toks1[np.in1d(toks1, [0, 3, 5, 7])] = 0
        toks1[np.in1d(toks1, [2, 4, 6, 8])] = 2
        toks2[np.in1d(toks2, [0, 3, 5, 7])] = 0
        toks2[np.in1d(toks2, [2, 4, 6, 8])] = 2

        p_spans = []
        span_start = -np.inf
        for i, ptok in enumerate(toks1):

            if ptok == 2:
                if span_start < -1: # else we merge the consecutive spans to avoid miscounting early finishes
                    span_start = i

            if ptok == 1 and span_start > -1:
                span_end = i
                p_spans.append((span_start, span_end))
                span_start = -np.inf

            if ptok == 0 and span_start < -1:
                    span_start = i-1

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
                error_counts[error_count_labels == 'wrong type'] += 1
                print('Span right, type wrong! ')
                print(preds1[idxs[0]:idxs[-1]+1])
                print(gt[idxs[0]:idxs[-1]+1])
                continue # not a mistake

            span_matched = False

            for pstart, pend in p_spans:
                if span_matched:
                    break

                if pstart > span_start and pstart < span_end:
                    if pstart == 0 and span_start < 0 and preds1[idxs[0]-1] != 1:
                        # it had already started the span
                        error_counts[error_count_labels == 'extra splits'] += 1
                    else:
                        # overlaps but starts late
                        error_counts[error_count_labels == 'late start'] += 1
                    span_matched = True
                elif pstart < span_start and pend > span_start:
                    if pstart == -1 and span_start >= 0 and gt[idxs[0]-1] != 1:
                        # the predicted span also covers a different gold span
                        error_counts[error_count_labels == 'fused together'] += 1
                    else:
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

    error_counts[error_count_labels == 'missed'] = np.sum(g_spans_with_error) - np.sum(g_span_partial_match) \
                                                   - error_counts[error_count_labels == 'wrong type']

    print(error_count_labels)
    print(error_counts)

    np.savetxt(output_path, analysis, delimiter=',', fmt='%i', header=analysis_header)
    np.savetxt(output_counts_path, error_counts[None, :], delimiter=',', fmt='%i', header=', '.join(error_count_labels))
    
if __name__ == '__main__':
    dataroot = os.path.expanduser('~/data/bayesian_sequence_combination/')

    # Analyse the errors introduced by our method that the baselines did not make.
    outroot = os.path.expanduser('~/data/bayesian_sequence_combination/output/')

    # NER ----------------------------------------------------------------------

    # prior_str = 'ner_task1_bac_ibcc_IF'
    # error_analysis(dataroot + '/data/ner/task1_test_gt.csv',
    #                dataroot + '/data/ner/task1_test_annos.csv',
    #                dataroot + '/data/ner/task1_test_doc_start.csv',
    #                outroot + '/from_krusty/ner-by-sentence_1/pred_started-2018-07-12-02-42-00-Nseen6056.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0)
    # prior_str = 'ner_task1_bac_vec_IF'
    # error_analysis(dataroot + '/data/ner/task1_test_gt.csv',
    #                dataroot + '/data/ner/task1_test_annos.csv',
    #                dataroot + '/data/ner/task1_test_doc_start.csv',
    #                outroot + '/from_krusty/ner-by-sentence_1/pred_started-2018-07-12-01-45-22-Nseen6056.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0)
    #
    # # prior_str = 'ner_task1_bac_seq_IF'
    # # error_analysis(dataroot + '/data/ner/task1_test_gt.csv',
    # #                dataroot + '/data/ner/task1_test_annos.csv',
    # #                dataroot + '/data/ner/task1_test_doc_start.csv',
    # #                outroot + '/ner/pred_started-2018-08-09-02-19-46-Nseen6056.csv',
    # #                outroot + '/analysis_%s' % prior_str,
    # #                outroot + '/analysis_counts_%s' % prior_str,
    # #                0)
    # #
    # # Analyse the errors that our method did not make but the baselines did.
    # prior_str = 'ner_task1_HMMCrowd'
    # error_analysis(dataroot + '/data/ner/task1_test_gt.csv',
    #                dataroot + '/data/ner/task1_test_annos.csv',
    #                dataroot + '/data/ner/task1_test_doc_start.csv',
    #                outroot + '/from_krusty/ner-by-sentence_1/pred_started-2018-07-06-22-24-18-Nseen6056.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0)
    #
    # # Analyse the errors that our method did not make but the baselines did.
    # prior_str = 'ner_task1_majority'
    # error_analysis(dataroot + '/data/ner/task1_test_gt.csv',
    #                dataroot + '/data/ner/task1_test_annos.csv',
    #                dataroot + '/data/ner/task1_test_doc_start.csv',
    #                outroot + '/ner/pred_started-2018-07-06-21-57-08-Nseen6056.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0)

    # PICO --------------------------------------------------------------------

    # prior_str = 'pico_task1_bac_ibcc_IF'
    # error_analysis(dataroot + '/data/bio/gt.csv',
    #                dataroot + '/data/bio/annos.csv',
    #                dataroot + '/data/bio/doc_start.csv',
    #                outroot + '/bio_task1/pred_started-2018-07-11-01-09-52-Nseen9480.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0, remove_val=True)
    #
    # prior_str = 'pico_task1_bac_ibcc_IF'
    # error_analysis(dataroot + '/data/bio/gt.csv',
    #                dataroot + '/data/bio/annos.csv',
    #                dataroot + '/data/bio/doc_start.csv',
    #                outroot + '/bio_task1/pred_started-2019-01-29-18-22-55-Nseen56858.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0, remove_val=True)

    # prior_str = 'pico_task1_bac_vec_IF'
    # error_analysis(dataroot + '/data/bio/gt.csv',
    #                dataroot + '/data/bio/annos.csv',
    #                dataroot + '/data/bio/doc_start.csv',
    #                outroot + '/bio_task1/pred_started-2018-06-13-01-43-11-Nseen9480.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0, remove_val=True)

    # prior_str = 'pico_task1_bac_seq_IF'
    # error_analysis(dataroot + '/data/bio/gt.csv',
    #                dataroot + '/data/bio/annos.csv',
    #                dataroot + '/data/bio/doc_start.csv',
    #                outroot + '/bio_task1/pred_started-2018-08-25-18-08-57-Nseen56858.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0, remove_val=True)

    # Analyse the errors that our method did not make but the baselines did.
    prior_str = 'pico_task1_HMMCrowd'
    error_analysis(dataroot + '/data/bio/gt.csv',
                   dataroot + '/data/bio/annos.csv',
                   dataroot + '/data/bio/doc_start.csv',
    #                outroot + '/bio_task1/pred_started-2018-08-25-23-55-53-Nseen56858.csv',
                   outroot + '/pico2/pred_started-2019-05-11-14-24-25-Nseen56858.csv',
                   outroot + '/analysis_%s' % prior_str,
                   outroot + '/analysis_counts_%s' % prior_str,
                   0, remove_val=True)

    # # Analyse the errors that our method did not make but the baselines did.
    # prior_str = 'pico_task1_majority'
    # error_analysis(dataroot + '/data/bio/gt.csv',
    #                dataroot + '/data/bio/annos.csv',
    #                dataroot + '/data/bio/doc_start.csv',
    #                outroot + '/bio_task1/pred_started-2018-08-29-20-26-38-Nseen56858.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0, remove_val=True)
    #
    # prior_str = 'pico_task1_bac_seq_IF_noLSTM'
    # error_analysis(dataroot + '/data/bio/gt.csv',
    #                dataroot + '/data/bio/annos.csv',
    #                dataroot + '/data/bio/doc_start.csv',
    #                outroot + '/pico/pred_started-2019-04-12-15-54-45-Nseen56858.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0, remove_val=True)
    #
    # prior_str = 'pico_task1_bac_seq_IF_noLSTM'
    # error_analysis(dataroot + '/data/bio/gt.csv',
    #                dataroot + '/data/bio/annos.csv',
    #                dataroot + '/data/bio/doc_start.csv',
    #                outroot + '/pico/pred_started-2019-04-11-14-18-37-Nseen56858.csv', #pred_started-2019-04-15-17-32-42-Nseen2134.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0, remove_val=True)
    #

    # prior_str = 'pico_task1_bac_seq_IF_noLSTM'
    # error_analysis(dataroot + '/data/bio/gt.csv',
    #                dataroot + '/data/bio/annos.csv',
    #                dataroot + '/data/bio/doc_start.csv',
    #                outroot + '/pico2/pred_started-2019-05-17-11-16-12-Nseen56858.csv', #pred_started-2019-04-15-17-32-42-Nseen2134.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0, remove_val=True)

    # prior_str = 'arg_MV'
    # error_analysis(dataroot + '/data/argmin_LMU/evaluation_gold.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_crowd.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_doc_start.csv',
    #                outroot + '/arg_LMU_final/pred_started-2019-05-11-00-25-58-Nseen40000.csv', #pred_started-2019-04-15-17-32-42-Nseen2134.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                0, remove_val=True)
    #
    # prior_str = 'arg_HMM_Crowd'
    # error_analysis(dataroot + '/data/argmin_LMU/evaluation_gold.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_crowd.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_doc_start.csv',
    #                outroot + '/arg_LMU_final/pred_started-2019-05-10-18-50-04-Nseen40000.csv', #pred_started-2019-04-15-17-32-42-Nseen2134.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                6, remove_val=True)
    #
    # prior_str = 'arg_bac_seq_IF'
    # error_analysis(dataroot + '/data/argmin_LMU/evaluation_gold.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_crowd.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_doc_start.csv',
    #                outroot + '/arg_LMU_final/pred_started-2019-05-09-23-33-51-Nseen40000.csv', #pred_started-2019-04-15-17-32-42-Nseen2134.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                7, remove_val=True)
    #
    # prior_str = 'arg_bac_CM_IF'
    # error_analysis(dataroot + '/data/argmin_LMU/evaluation_gold.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_crowd.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_doc_start.csv',
    #                outroot + '/arg_LMU_final/pred_started-2019-05-10-21-14-06-Nseen40000.csv', #pred_started-2019-04-15-17-32-42-Nseen2134.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                1, remove_val=True)
    #
    # prior_str = 'arg_bac_VEC_IF'
    # error_analysis(dataroot + '/data/argmin_LMU/evaluation_gold.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_crowd.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_doc_start.csv',
    #                outroot + '/arg_LMU_final/pred_started-2019-05-10-22-26-46-Nseen40000.csv', #pred_started-2019-04-15-17-32-42-Nseen2134.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                2, remove_val=True)
    #
    # prior_str = 'arg_IBCC'
    # error_analysis(dataroot + '/data/argmin_LMU/evaluation_gold.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_crowd.csv',
    #                dataroot + '/data/argmin_LMU/evaluation_doc_start.csv',
    #                outroot + '/arg_LMU_final/pred_started-2019-05-19-01-47-32-Nseen40000.csv', #pred_started-2019-04-15-17-32-42-Nseen2134.csv',
    #                outroot + '/analysis_%s' % prior_str,
    #                outroot + '/analysis_counts_%s' % prior_str,
    #                3, remove_val=True)