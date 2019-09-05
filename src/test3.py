'''
Like test 1 but we add in some training data

SEMI-SUPERVISED: use a little bit of training data from the target domain --------------------------------------------

'''
import os

import numpy as np
import json

from baselines.dawid_and_skene import ibccvb
from base_models import run_base_models
from bsc import bsc
from helpers import evaluate, Dataset, get_anno_matrix, get_root_dir, append_training_labels

reload = True
rerun_aggregators = True
verbose = False

datadir = os.path.join(get_root_dir(), 'data/famulus_TEd')

nclasses = 9 # four types means I and B tags for each type + 1 O tag gives 9 different tags or class labels

base_models = ['bilstm-crf', 'crf']#, 'flair-pos']#, 'flair-ner'] # 'flair' -- not implemented yet, do this if we can't get lample to work

print('Using base models: ' + str(base_models))

means_ibcc = []
means_bsc = []
means_every = []
means_mv = []

for classid in [0, 1, 2, 3]:

    basemodels_str = '--'.join(base_models)

    resdir = os.path.join(get_root_dir(), 'output/famulus_TEd_task3_type%i_basemodels%s' % (classid, '--'.join(base_models)) )
    if not os.path.exists(resdir):
        os.mkdir(resdir)

    predfile = os.path.join(resdir, 'preds.json')
    trpredfile = os.path.join(resdir, 'trpreds.json')
    resfile = os.path.join(resdir, 'res.json')

    if reload and not rerun_aggregators and os.path.exists(predfile):
        with open(predfile, 'r') as fh:
            preds = json.load(fh)

        with open(trpredfile, 'r') as fh:
            trpreds = json.load(fh)

        with open(resfile, 'r') as fh:
            res = json.load(fh)
    else:
        preds = {}
        trpreds = {}
        res = {}

    for base_model_str in base_models:

        dataset = Dataset(datadir, classid)
        basepreds, basetrpreds, baseres = run_base_models(dataset, classid, base_model_str, reload)
        for key in basepreds:

            if key == 'a' or key == 'MV' or key == 'baseline_every' or 'ibcc' in key or 'bsc-seq' in key:
                continue # 'a' is the in-domain performance. We don't use the in-domain model as part of an ensemble.
                # The others are crap that shouldn't be in there.

            if len(basepreds[key]) == 0:
                continue # skip any entries that don't really have predictions. Why are they there?

            ntest_domains = len(basepreds[key])

            if verbose:
                print('Processing model type %s, base labeller %s we found %i sets of test results' %
                  (base_model_str, key, ntest_domains))

            new_key = base_model_str + '_' + str(classid) + '_' + key

            preds[new_key] = basepreds[key]
            trpreds[new_key] = basetrpreds[key]

    preds['agg_ibcc'] = []
    preds['agg_bsc-seq'] = []

    res['agg_ibcc'] = []
    res['agg_bsc-seq'] = []
    res['baseline_every'] = []
    res['agg_MV'] = []

    # Now we do a simulation with different amounts of data from the target domain
    for didx, tedomain in enumerate(dataset.domains):

        N = len(dataset.tetext[tedomain]) # number of test points
        # First, put the base labellers into a table.

        batchsize = 10
        nbatches = 6

        preds['agg_ibcc'].append([])
        res['agg_ibcc'].append([])

        preds['agg_bsc-seq'].append([])
        res['agg_bsc-seq'].append([])

        basepreds, basetrpreds, baseres = run_base_models(dataset, classid, base_model_str, reload)
        res_every = evaluate(basepreds['baseline_every'][didx], dataset.tegold[tedomain], dataset.tedocstart[tedomain])
        res['baseline_every'].append([])

        votes, _ = get_anno_matrix(classid, preds, didx)
        counts = np.zeros((votes.shape[0], 3))
        for c in range(3): # iterate through IOB
            counts[:, c] = np.sum(votes==c, axis=1)

        mv = np.argmax(counts, axis=1)
        res_mv = evaluate(mv, dataset.tegold[tedomain], dataset.tedocstart[tedomain])
        res['agg_MV'].append([])

        # take the training labels for aggregation methods from the target domain's training set.
        # -- this means that the test set for the target domain stays the same.
        for b in range(0, nbatches):

            res['baseline_every'][didx].append(res_every)
            res['agg_MV'][didx].append(res_mv)

            # get the training labels for the training set
            annos, uniform_priors = get_anno_matrix(classid, preds, didx, include_all=False)
            if b == 0:
                docstart = dataset.tedocstart[tedomain]
                text = dataset.tetext[tedomain]
                trlabels = np.zeros(N) - 1
            else:
                annos, docstart, text, trlabels = append_training_labels(annos, basemodels_str, dataset, classid, didx, tedomain, trpreds, 10)
            K = annos.shape[1]  # number of annotators

            max_iter = 30
            alpha0_factor = 0.1
            alpha0_diags = 0.1
            nu0_factor = 0.1

            # now run IBCC
            probs, _ = ibccvb(annos, nclasses, nu0_factor, alpha0_factor, alpha0_diags, max_iter, trlabels)
            agg = np.argmax(probs, axis=1)
            agg = agg[:N] # ignore the training items
            preds['agg_ibcc'][didx].append(agg.flatten().tolist())

            res_s = evaluate(agg, dataset.tegold[tedomain], dataset.tedocstart[tedomain])
            res['agg_ibcc'][didx].append(res_s)
            print('F1 score=%s for IBCC with %i training docs on %s' % (str(res_s), b*batchsize, tedomain))

            # -------------------------------------------------------------------------------------------------------------

            alpha0_factor = 0.1
            alpha0_diags = 10
            nu0_factor = 0.1

            # # now run BSC-seq
            bsc_model = bsc.BSC(L=nclasses, K=K, max_iter=max_iter, before_doc_idx=1, inside_labels=[0],
                            outside_labels=[1], beginning_labels=[2],
                            alpha0_diags=alpha0_diags, alpha0_factor=alpha0_factor, beta0_factor=nu0_factor,
                            worker_model='seq', tagging_scheme='IOB2', data_model=[], transition_model='HMM',
                            no_words=False)

            bsc_model.verbose = False
            bsc_model.max_internal_iters = max_iter

            probs, agg, pseq = bsc_model.run(annos, docstart, text, converge_workers_first=False, gold_labels=trlabels)
            agg = agg[:N] # forget the training points
            preds['agg_bsc-seq'][didx].append(agg.flatten().tolist())

            res_s = evaluate(agg, dataset.tegold[tedomain], dataset.tedocstart[tedomain])
            res['agg_bsc-seq'][didx].append(res_s)

            print('F1 score=%s for BSC-seq with %i training docs on %s' % (str(res_s), b*batchsize, tedomain))
            # -------------------------------------------------------------------------------------------------------------

    means_ibcc.append(np.mean(np.array(res['agg_ibcc']), axis=0))
    means_bsc.append(np.mean(np.array(res['agg_bsc-seq']), axis=0))
    means_every.append(np.mean(np.array(res['baseline_every']), axis=0))
    means_mv.append(np.mean(np.array(res['agg_MV']), axis=0))

    print('Average F1 scores for IBCC with increasing amounts of training data:')
    print(means_ibcc[-1])

    print('Average F1 scores for BSC-seq with increasing amounts of training data:')
    print(means_bsc[-1])

    print('Average F1 scores for baseline_every with increasing amounts of training data:')
    print(means_every[-1])

    # save all the new results
    with open(predfile, 'w') as fh:
        json.dump(preds, fh)

    with open(trpredfile, 'w') as fh:
        json.dump(trpreds, fh)

    with open(resfile, 'w') as fh:
        json.dump(res, fh)

# plot results
import matplotlib.pyplot as plt

plt.figure()
plt.plot(np.mean(means_ibcc, axis=0), label='IBCC')
plt.plot(np.mean(means_bsc, axis=0), label='BSC-seq')
plt.plot(np.mean(means_every, axis=0), label='BiLSTM-LSTM-CRF')
plt.plot(np.mean(means_mv, axis=0), label='MV')
plt.xlabel('Number of training docs in target domain')
plt.ylabel('F1')
plt.legend(loc='best')

plt.savefig('results/test3.pdf')