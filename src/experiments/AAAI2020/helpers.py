import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

from evaluation.span_level_f1 import f1


def get_root_dir():
    # return os.path.expanduser('~/data/bayesian_sequence_combination/')
    return './data/famulus/'


def evaluate(preds_all_types, gold_all_types, doc_start, spantype=0, f1type='tokens', detailed=False):

    preds_all_types = np.array(preds_all_types)

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
        preds = np.array(preds_all_types)

    # only consider the points where there are real prediction
    gold = gold[preds != -1]
    preds = preds[preds != -1]

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

        annos.append(np.array(preds[base][didx])[:, None])

        if include_all and int(base.split('_')[1]) != spantype:
            uniform_priors.append(counter)

        counter += 1

    annos = np.concatenate(annos, axis=1)

    return annos, uniform_priors

def append_training_labels(annos, basemodels_str, dataset, classid, didx, tedomain, trpreds, Ntrain):

    N = len(dataset.tetext[tedomain])  # number of test points

    if 'bilstm-crf' in basemodels_str:
        otheridx = didx + 1 if (didx < len(dataset.domains) - 2) else (didx - 1)

        for basemodel_str in basemodels_str.split('--'):

            othername = '%s_%i_' % (basemodel_str, classid) + dataset.domains[otheridx]
            newname = '%s_%i_' % (basemodel_str, classid) + tedomain

            if othername not in trpreds:
                othername = '%s_%i__' % (basemodel_str, classid) + dataset.domains[otheridx]
                newname = '%s_%i__' % (basemodel_str, classid) + tedomain

            Nother = len(trpreds[othername][didx])
            trpreds[newname][didx] = (np.zeros(Nother) - 1).tolist()

    trannos, _ = get_anno_matrix(classid, trpreds, didx, include_all=False)

    # get the training labels for the training set
    Ndoc = np.sum(dataset.trdocstart[tedomain])
    docids = np.cumsum(dataset.trdocstart[tedomain])

    trdocs = np.random.choice(Ndoc, Ntrain, replace=False)
    tridxs = np.in1d(docids, trdocs)
    trlabels = dataset.trgold[tedomain][tridxs]
    trlabels = np.concatenate(
        (np.zeros(N) - 1, trlabels))  # concatenate with a vector of missing training labels for the test items
    annos = np.concatenate((annos, trannos[tridxs]), axis=0)

    docstart = np.concatenate((dataset.tedocstart[tedomain], dataset.trdocstart[tedomain][tridxs]))
    text = np.concatenate((dataset.tetext[tedomain], dataset.trtext[tedomain][tridxs]))

    return annos, docstart, text, trlabels


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
                               names=['features', 'gold', 'doc_start'])
            self.trgold[subset] = to_IOB(data['gold'].values)
            self.trtext[subset] = data['features'].values
            self.trdocstart[subset] = data['doc_start'].values

            data = pd.read_csv(os.path.join(subsetdir, 'eda_dev_ES.tsv'), sep='\t', usecols=[0, goldcol, 11],
                               names=['features', 'gold', 'doc_start'])
            self.degold[subset] = to_IOB(data['gold'].values)
            self.detext[subset] = data['features'].values
            self.dedocstart[subset] = data['doc_start'].values

            data = pd.read_csv(os.path.join(subsetdir, 'eda_test_ES.tsv'), sep='\t', usecols=[0, goldcol, 11],
                               names=['features', 'gold', 'doc_start'])
            self.tegold[subset] = to_IOB(data['gold'].values)
            self.tetext[subset] = data['features'].values
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
