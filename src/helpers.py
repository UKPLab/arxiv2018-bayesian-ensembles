import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

from evaluation.span_level_f1 import f1

def get_root_dir():
    # return os.path.expanduser('~/data/bayesian_sequence_combination/')
    return './data/famulus/'

def evaluate(preds_all_types, gold_all_types, doc_start, spantype=0, f1type='tokens', detailed=False):

    gold = np.ones(len(gold_all_types))
    gold[gold_all_types == (spantype * 2 + 2)] = 2

    if spantype == 0:
        gold[gold_all_types == 0] = 0
    else:
        gold[gold_all_types == (spantype * 2 + 1)] = 0

    if np.max(preds_all_types) > 2:
        # we also need to split the types for the predictions
        preds = np.ones(len(preds_all_types))
        preds[preds_all_types == (spantype * 2 + 2)] = 2
        if spantype == 0:
            preds[preds_all_types == 0] = 0
        else:
            preds[preds_all_types == (spantype * 2 + 1)] = 0
    else:
        preds = preds_all_types

    if f1type == 'all':
        if detailed:
            print('Precision for classes I, O, B: ' + str(np.around(precision_score(gold, preds, average=None), 2)))
            print('Recall for classes I, O, B:    ' + str(np.around(recall_score(gold, preds, average=None), 2)))
            print('I to B confusion: %i' % np.sum((gold == 0) & (preds == 2)) )
            print('I to O confusion: %i' % np.sum((gold == 0) & (preds == 1)))
            print('B to I confusion: %i' % np.sum((gold == 2) & (preds == 0)))
            print('B to O confusion: %i' % np.sum((gold == 2) & (preds == 1)))
            print('O to I confusion: %i' % np.sum((gold == 1) & (preds == 0)))
            print('O to B confusion: %i' % np.sum((gold == 1) & (preds == 2)))

            print('Correct I: %i' % np.sum((gold == 0) & (preds == 0)))
            print('Correct O: %i' % np.sum((gold == 1) & (preds == 1)))
            print('Correct B: %i' % np.sum((gold == 2) & (preds == 2)))

        return [
            f1_score(gold, preds, average='macro'),
            f1_score((gold==0)|(gold==2), (preds==0)|(preds==2)),
            f1(preds, gold, True, doc_start),
            f1(preds, gold, False, doc_start),
        ]
    elif f1type == 'tokens':
        return f1_score(gold, preds, average='macro')


def get_anno_names(spantype, preds, include_all=False):

    names = []

    for base in preds:
        if '_every' in base or base == 'a' or 'agg_' in base or 'pos' in base or 'ner' in base or 'baseline_z' in base:
            continue  # exclude base labellers trained on the target domain or aggregation results
            # exclude other predictors that did not label the target classes

        if not include_all and int(base.split('_')[1]) != spantype:
            continue

        names.append(base)

    return names

def get_anno_matrix(spantype, preds, didx, include_all=False):
    annos = []
    uniform_priors = []

    counter = 0
    # print('Creating annotator matrix:')
    for base in preds:
        if '_every' in base or base == 'a' or 'agg_' in base or 'pos' in base or 'ner' in base or 'baseline_z' in base:
            continue  # exclude base labellers trained on the target domain or aggregation results
            # exclude other predictors that did not label the target classes

        if not include_all and int(base.split('_')[1]) != spantype:
            continue

        # print('Included %s' % base)
        annos.append(np.array(preds[base][didx])[:, None])

        if include_all and int(base.split('_')[1]) != spantype:
            uniform_priors.append(counter)

        counter += 1

    annos = np.concatenate(annos, axis=1)

    return annos, uniform_priors


class Dataset:

    def __init__(self, datadir, classid, verbose=False):

        print('loading data from %s' % datadir)

        subsets = sorted(os.listdir(datadir))

        self.trgold = {}
        self.trtext = {}
        self.trdocstart = {}

        self.tegold = {}
        self.tetext = {}
        self.tedocstart = {}

        self.degold = {}
        self.detext = {}
        self.dedocstart = {}

        # For each fold, load data
        self.domains = []

        totaldocs = 0

        for subset in subsets:

            subsetdir = os.path.join(datadir, subset)

            if not os.path.isdir(subsetdir):
                continue
            elif 'Case' not in subset:
                continue
            else:
                self.domains.append(subset)

            goldcol = classid + 6

            def to_IOB(labels):
                labels[(np.mod(labels, 2) == 0) & (labels != 0)] = 2
                labels[(np.mod(labels, 2) == 1) & (labels != 1)] = 0

                return labels

            data = pd.read_csv(os.path.join(subsetdir, 'eda_train_ES.tsv'), sep='\t', usecols=[0, goldcol, 11],
                               names=['text', 'gold', 'doc_start'])
            self.trgold[subset] = to_IOB(data['gold'].values)
            self.trtext[subset] = data['text'].values
            self.trdocstart[subset] = data['doc_start'].values

            data = pd.read_csv(os.path.join(subsetdir, 'eda_dev_ES.tsv'), sep='\t', usecols=[0, goldcol, 11],
                               names=['text', 'gold', 'doc_start'])
            self.degold[subset] = to_IOB(data['gold'].values)
            self.detext[subset] = data['text'].values
            self.dedocstart[subset] = data['doc_start'].values

            data = pd.read_csv(os.path.join(subsetdir, 'eda_test_ES.tsv'), sep='\t', usecols=[0, goldcol, 11],
                               names=['text', 'gold', 'doc_start'])
            self.tegold[subset] = to_IOB(data['gold'].values)
            self.tetext[subset] = data['text'].values
            self.tedocstart[subset] = data['doc_start'].values

            Ntargettr = len(self.trdocstart[subset])
            Ndoc = np.sum(self.trdocstart[subset])
            totaldocs += Ndoc
            if verbose:
                print('LOADING: Domain %s has %i training docs and %i training tokens' % (subset, Ndoc, Ntargettr))

            Ntargetde = len(self.dedocstart[subset])
            Ndoc = np.sum(self.dedocstart[subset])
            totaldocs += Ndoc
            if verbose:
                print('LOADING: Domain %s has %i dev docs and %i tokens' % (subset, Ndoc, Ntargetde))

            Ntargette = len(self.tedocstart[subset])
            Ndoc = np.sum(self.tedocstart[subset])
            totaldocs += Ndoc
            if verbose:
                print('LOADING: Domain %s has %i test docs and %i tokens' % (subset, Ndoc, Ntargette))

        if verbose:
            print('LOADING: Total number of documents in the dataset: %i' % totaldocs)
