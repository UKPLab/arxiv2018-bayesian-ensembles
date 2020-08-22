'''
Like test 1 but we add in some training data

SEMI-SUPERVISED learning on FAMULUS: use a little bit of training data from the target domain --------------------------------------------

'''
import os

import numpy as np
import json

from baselines.dawid_and_skene import ibccvb
from evaluation.experiment import output_root_dir, data_root_dir
from experiments.AAAI2020.base_models import run_base_models
from bayesian_combination import bayesian_combination
from experiments.AAAI2020.helpers import evaluate, Dataset, get_anno_matrix, get_root_dir, append_training_labels

reload = False
verbose = False
rerun = False

# if len(sys.argv) > 1:
#     uberdomain = sys.argv[1]
# else:
#     uberdomain = 'TEd'

batchsize = 10
nbatches = 6

base_models = ['bilstm-crf', 'crf']#, 'flair-pos']#, 'flair-ner'] # 'flair' -- not implemented yet, do this if we can't get lample to work

print('Using base models: ' + str(base_models))

def run_test3(uberdomain):
    datadir = os.path.join(get_root_dir(), 'data/famulus_%s' % uberdomain)

    means_ibcc = []
    means_bsc = []
    means_every = []
    means_mv = []

    for classid in [0, 1, 2, 3]:

        basemodels_str = '--'.join(base_models)

        resdir = os.path.join(output_root_dir, 'famulus_%s_task3_type%i_basemodels%s'
                              % (uberdomain, classid, '--'.join(base_models)) )
        if not os.path.exists(resdir):
            os.mkdir(resdir)

        predfile = os.path.join(resdir, 'preds.json')
        trpredfile = os.path.join(resdir, 'trpreds.json')
        resfile = os.path.join(resdir, 'res.json')

        if reload and os.path.exists(predfile):
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
            basepreds, basetrpreds, baseres = run_base_models(dataset, classid, uberdomain, base_model_str, reload)
            for key in basepreds:

                if key == 'a' or key == 'MV' or key == 'baseline_every' or 'ibcc' in key or 'bayesian_combination-seq' in key:
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

        if 'agg_ibcc' not in preds or rerun:
            preds['agg_ibcc'] = []

        if 'agg_bsc-seq' not in preds or rerun:
            preds['agg_bsc-seq'] = []
        else:
            preds['agg_bsc-seq'] = preds['agg_bsc-seq'][:8]
        preds['baseline_every'] = []
        preds['agg_MV'] = []

        res['agg_ibcc'] = []
        res['agg_bsc-seq'] = []
        res['baseline_every'] = []
        res['agg_MV'] = []

        allgold = []
        alldocstart = []

        # Now we do a simulation with different amounts of data from the target domain
        for didx, tedomain in enumerate(dataset.domains):

            allgold.append(dataset.tegold[tedomain])
            alldocstart.append(dataset.tedocstart[tedomain])

            N = len(dataset.tetext[tedomain]) # number of test points
            # First, put the base labellers into a table.

            if len(preds['agg_ibcc']) <= didx or rerun:
                preds['agg_ibcc'].append([])

            if len(preds['agg_bsc-seq']) <= didx or rerun:
                preds['agg_bsc-seq'].append([])

            if didx == 0:
                basepreds, basetrpreds, baseres = run_base_models(dataset, classid, uberdomain, base_model_str, reload)
            preds['baseline_every'].append([])

            votes, _ = get_anno_matrix(classid, preds, didx)
            counts = np.zeros((votes.shape[0], 3))
            for c in range(3): # iterate through IOB
                counts[:, c] = np.sum(votes==c, axis=1)

            mv = np.argmax(counts, axis=1)
            preds['agg_MV'].append([])

            # take the training labels for aggregation methods from the target domain's training set.
            # -- this means that the test set for the target domain stays the same.
            for b in range(0, nbatches):

                preds['baseline_every'][didx].append(basepreds['baseline_every'][didx])
                preds['agg_MV'][didx].append(mv.tolist())

                # get the training labels for the training set
                annos, uniform_priors = get_anno_matrix(classid, preds, didx, include_all=False)
                if b == 0:
                    docstart = dataset.tedocstart[tedomain]
                    text = dataset.tetext[tedomain]
                    trlabels = np.zeros(N) - 1
                else:
                    annos, docstart, text, trlabels = append_training_labels(annos, basemodels_str, dataset, classid, didx, tedomain, trpreds, batchsize)
                K = annos.shape[1]  # number of annotators

                max_iter = 100
                alpha0_factor = 0.1
                alpha0_diags = 0.1
                beta0_factor = 0.1

                # now run IBCC
                if rerun or 'agg_ibcc' not in preds or len(preds['agg_ibcc'][didx]) <= b:
                    probs, _ = ibccvb(annos, 3, beta0_factor, alpha0_factor, alpha0_diags, max_iter=max_iter,
                                      target_labels=trlabels)
                    agg = np.argmax(probs, axis=1)
                    agg = agg[:N] # ignore the training items
                    preds['agg_ibcc'][didx].append(agg.flatten().tolist())

                # -------------------------------------------------------------------------------------------------------------

                alpha0_factor = 0.1
                alpha0_diags = 10
                beta0_factor = 0.1

                if rerun or 'agg_bsc-seq' not in preds or len(preds['agg_bsc-seq'][didx]) <= b:
                    # # now run BSC-seq
                    bsc_model = bayesian_combination.BC(L=3, K=K, max_iter=max_iter, outside_label=1,
                                alpha0_diags=alpha0_diags, alpha0_factor=alpha0_factor, beta0_factor=beta0_factor,
                                annotator_model='seq', tagging_scheme='IOB2', taggers=[], true_label_model='HMM',
                                discrete_feature_likelihoods=True, converge_workers_first=False)

                    bsc_model.verbose = False
                    bsc_model.max_internal_iters = max_iter

                    probs, agg, pseq = bsc_model.fit_predict(annos, docstart, text, gold_labels=trlabels)
                    agg = agg[:N] # forget the training points
                    preds['agg_bsc-seq'][didx].append(agg.flatten().tolist())

                # -------------------------------------------------------------------------------------------------------------

        allgold = np.concatenate(allgold)
        alldocstart = np.concatenate(alldocstart)

        res['agg_ibcc'] = []
        res['agg_bsc-seq'] = []
        res['baseline_every'] = []
        res['agg_MV'] = []

        for b in range(nbatches):
            res['agg_ibcc'].append(evaluate(np.concatenate(np.array(preds['agg_ibcc'])[:, b], axis=0), allgold, alldocstart, f1type='all'))
            res['agg_bsc-seq'].append(evaluate(np.concatenate(np.array(preds['agg_bsc-seq'])[:, b], axis=0), allgold, alldocstart, f1type='all'))
            res['baseline_every'].append(evaluate(np.concatenate(np.array(preds['baseline_every'])[:, b], axis=0), allgold, alldocstart, f1type='all'))
            res['agg_MV'].append(evaluate(np.concatenate(np.array(preds['agg_MV'])[:, b], axis=0), allgold, alldocstart, f1type='all'))

        means_ibcc.append(res['agg_ibcc'])
        means_bsc.append(res['agg_bsc-seq'])
        means_every.append(res['baseline_every'])
        means_mv.append(res['agg_MV'])

        print('Average F1 scores for IBCC with increasing amounts of training data:')
        print(means_ibcc[-1])

        print('Average F1 scores for BSC-seq with increasing amounts of training data:')
        print(means_bsc[-1])

        print('Average F1 scores for baseline_every with increasing amounts of training data:')
        print(means_every[-1])

        # save all the new results
        # with open(predfile, 'w') as fh:
        #     json.dump(preds, fh)
        #
        # with open(trpredfile, 'w') as fh:
        #     json.dump(trpreds, fh)
        #
        # with open(resfile, 'w') as fh:
        #     json.dump(res, fh)

    return means_ibcc, means_bsc, means_every, means_mv

means_ibcc, means_bsc, means_every, means_mv = run_test3('MEd')
means_ibcc2, means_bsc2, means_every2, means_mv2 = run_test3('TEd')

for classid in range(4):
    # plot results
    import matplotlib.pyplot as plt

    plt.figure(figsize=[2.5,3])

    # take the mean across the four different classes
    plt.plot(np.arange(nbatches)*batchsize, 100 * np.mean([means_ibcc[classid], means_ibcc2[classid]], axis=0)[:, 1][:nbatches], label='IBCC', marker='>', linestyle=':')
    plt.plot(np.arange(nbatches)*batchsize, 100 * np.mean([means_bsc[classid], means_bsc2[classid]], axis=0)[:, 1][:nbatches], label='BSC-seq', marker='o', linestyle='-.')
    plt.plot(np.arange(nbatches)*batchsize, 100 * np.mean([means_every[classid], means_every2[classid]], axis=0)[:, 1][:nbatches], label='All-domain CRF', marker='x', linestyle='--')
    plt.plot(np.arange(nbatches)*batchsize, 100 * np.mean([means_mv[classid], means_mv2[classid]], axis=0)[:, 1][:nbatches], label='MV', marker='D', linestyle='-')
    plt.xlabel('Num. training sentences')
    plt.ylabel('F1 token-wise')
    plt.legend(loc='best')
    #plt.ylim(0,80)
    plt.tight_layout()

    if not os.path.exists(output_root_dir + '/famulus_results'):
        os.mkdir(output_root_dir + '/famulus_results')

    plt.savefig(output_root_dir + '/famulus_results/test3_tokf1_%i.pdf' % classid)

    plt.figure(figsize=[2.5,3])
    plt.plot(np.arange(nbatches)*batchsize, 100 * np.mean([means_ibcc[classid], means_ibcc2[classid]], axis=0)[:, 3][:nbatches], label='IBCC', marker='>', linestyle=':')
    plt.plot(np.arange(nbatches)*batchsize, 100 * np.mean([means_bsc[classid], means_bsc2[classid]], axis=0)[:, 3][:nbatches], label='BSC-seq', marker='o', linestyle='-.')
    plt.plot(np.arange(nbatches)*batchsize, 100 * np.mean([means_every[classid], means_every2[classid]], axis=0)[:, 3][:nbatches], label='All-domain CRF', marker='x', linestyle='--')
    plt.plot(np.arange(nbatches)*batchsize, 100 * np.mean([means_mv[classid], means_mv2[classid]], axis=0)[:, 3][:nbatches], label='MV', marker='D', linestyle='-')
    plt.xlabel('Num. training sentences in target domain')
    plt.ylabel('F1 relaxed spans')
    plt.legend(loc='lower center')
    #plt.ylim(0,80)
    plt.tight_layout()

    plt.savefig(output_root_dir + '/famulus_results/test3_spanf1_%i.pdf' % classid)