'''
Created on Jul 11, 2017

@author: Melvin Laux
'''

import numpy as np
from sklearn import metrics as skm

from data import data_utils
from evaluation.plots import SCORE_NAMES
from evaluation.span_level_f1 import strict_span_metrics_2, precision, recall, f1

'''
Calculates the absolute difference of numbers of found annos between the prediction and the target sequence. 
This is done by counting the number of 'B' tokens in each sequence. 
'''        
def count_error(pred, target):
    return len(pred==2) - len(target==2)

'''
Calculates the number of invalid labels in the given prediction, i.e. the number of 'I' tokens directly following an 'O' token.
'''        
def num_invalid_labels(pred, doc_start, restricted=[0, 3, 5, 7], outside=1):

    docs = np.split(pred, np.where(doc_start==1)[0])[1:]
    try:
        count_invalid = lambda doc: len(np.where(np.in1d(doc[np.r_[0, np.where(doc[:-1] == outside)[0] + 1]], restricted))[0])
    except:
        print('Could not count any invalid labels because some classes were not present.')
        count_invalid = 0
    return np.sum(list(map(count_invalid, docs))) #/float(len(pred))

'''
Calculates the difference of the mean length of annos between prediction and target sequences.
'''        
def mean_length_error(pred, target, doc_start):
    
    get_annos = lambda x: np.split(x, np.union1d(np.where(doc_start==1)[0], np.where(x==2)[0]))
    
    del_o = lambda x: np.delete(x, np.where(x==1)[0])
    
    not_empty = lambda x: np.size(x) > 0
    
    mean_pred = np.mean(np.array([len(x) for x in filter(not_empty, list(map(del_o, get_annos(pred))))]))
    if mean_pred == np.nan:
        mean_pred = 0
    mean_target = np.mean(np.array([len(x) for x in filter(not_empty, list(map(del_o, get_annos(target))))]))

    return np.abs(mean_pred-mean_target)

if __name__ == '__main__':
    pass


def calculate_sample_metrics(nclasses, agg, gt, probs, doc_starts, print_per_class_results=False):
    result = -np.ones(len(SCORE_NAMES) - 3)

    gt = gt.astype(int)

    probs[np.isnan(probs)] = 0
    probs[np.isinf(probs)] = 0

    # token-level metrics
    result[0] = skm.accuracy_score(gt, agg)

    # the results are undefined if some classes are not present in the gold labels
    prec_by_class = skm.precision_score(gt, agg, average=None, labels=range(nclasses))
    rec_by_class = skm.recall_score(gt, agg, average=None, labels=range(nclasses))
    f1_by_class = skm.f1_score(gt, agg, average=None, labels=range(nclasses))

    if print_per_class_results:
        print('Token Precision:')
        print(prec_by_class)

        print('Token Recall:')
        print(rec_by_class)

        print('Token F1:')
        print(f1_by_class)

    result[1] = np.mean(prec_by_class[np.unique(gt)])
    result[2] = np.mean(rec_by_class[np.unique(gt)])
    result[3] = np.mean(f1_by_class[np.unique(gt)])

    # span-level metrics - strict
    p, r, f = strict_span_metrics_2(agg, gt, doc_starts)
    result[6] = p  # precision(agg, gt, True, doc_starts)
    result[7] = r  # recall(agg, gt, True, doc_starts)
    result[8] = f  # f1(agg, gt, True, doc_starts)

    # span-level metrics -- relaxed
    result[9] = precision(agg, gt, False, doc_starts)
    result[10] = recall(agg, gt, False, doc_starts)
    result[11] = f1(agg, gt, False, doc_starts)

    auc_score = 0
    total_weights = 0
    for i in range(probs.shape[1]):

        if not np.any(gt == i) or np.all(gt == i) or np.any(np.isnan(probs[:, i])) or np.any(np.isinf(probs[:, i])):
            print('Could not evaluate AUC for class %i -- all data points have same value.' % i)
            continue

        auc_i = skm.roc_auc_score(gt == i, probs[:, i])
        # print 'AUC for class %i: %f' % (i, auc_i)
        auc_score += auc_i * np.sum(gt == i)
        total_weights += np.sum(gt == i)

        if print_per_class_results:
            print('AUC for class %i = %f' % (i, auc_i))

    result[4] = auc_score / float(total_weights) if total_weights > 0 else 0

    result[5] = skm.log_loss(gt, probs, eps=1e-100, labels=np.arange(nclasses))

    return result


def calculate_scores(postprocess, agg, gt, probs, doc_start, bootstrapping=True, print_per_class_results=False):
    result = -np.ones((len(SCORE_NAMES), 1))

    # exclude data points with missing ground truth
    agg = agg[gt != -1]
    probs = probs[gt != -1]
    doc_start = doc_start[gt != -1]
    gt = gt[gt != -1]

    print('unique ground truth values: ')
    print(np.unique(gt))
    print('unique prediction values: ')
    print(np.unique(agg))

    if postprocess:
        agg = data_utils.postprocess(agg, doc_start)

    print('Plotting confusion matrix for errors: ')

    nclasses = probs.shape[1]
    conf = np.zeros((nclasses, nclasses))
    for i in range(nclasses):
        for j in range(nclasses):
            conf[i, j] = np.sum((gt == i).flatten() & (agg == j).flatten())

        # print 'Acc for class %i: %f' % (i, skm.accuracy_score(gt==i, agg==i))
    print(np.round(conf))

    gold_doc_start = np.copy(doc_start)
    gold_doc_start[gt == -1] = 0

    Ndocs = int(np.sum(gold_doc_start))
    if Ndocs < 100 and bootstrapping:

        print('Using bootstrapping with small test set size')

        # use bootstrap resampling with small datasets
        nbootstraps = 100
        nmetrics = len(SCORE_NAMES) - 3
        resample_results = np.zeros((nmetrics, nbootstraps))

        for i in range(nbootstraps):
            print('Bootstrapping the evaluation: %i of %i' % (i, nbootstraps))
            not_sampled = True
            nsample_attempts = 0
            while not_sampled:
                sampleidxs = np.random.choice(Ndocs, Ndocs, replace=True)
                sampleidxs = np.in1d(np.cumsum(gold_doc_start) - 1, sampleidxs)

                if len(np.unique(gt[sampleidxs])) >= 2:
                    not_sampled = False
                    # if we have at least two class labels in the sample ground truth, we can use this sample

                nsample_attempts += 1
                if nsample_attempts >= Ndocs:
                    not_sampled = False

            if nsample_attempts <= Ndocs:
                resample_results[:, i] = calculate_sample_metrics(nclasses,
                                                                  agg[sampleidxs],
                                                                  gt[sampleidxs],
                                                                  probs[sampleidxs],
                                                                  doc_start[sampleidxs],
                                                                  print_per_class_results)
            else:  # can't find enough valid samples for the bootstrap, let's just use what we got so far.
                resample_results = resample_results[:, :i]
                break

        sample_res = np.mean(resample_results, axis=1)
        result[:len(sample_res), 0] = sample_res
        std_result = np.std(resample_results, axis=1)

    else:
        sample_res = calculate_sample_metrics(nclasses, agg, gt, probs, doc_start, print_per_class_results)
        result[:len(sample_res), 0] = sample_res

        std_result = None

    result[len(sample_res)] = count_error(agg, gt)
    result[len(sample_res) + 1] = num_invalid_labels(agg, doc_start)
    result[len(sample_res) + 2] = mean_length_error(agg, gt, doc_start)

    print('CEE tokens = %f' % result[5])
    print('F1 score tokens = %f' % result[3])
    print('F1 score spans strict = %f' % result[8])
    print('F1 score spans relaxed = %f' % result[11])

    return result, std_result