'''

Hypothesis 1: ensemble learning with BCC improves performance with no training data.


*Plot* the F1 score with increasing amounts of training data for BCC. Tested methods:
(BASELINE x) majority vote;
(BASELINE y) best individual source model performance
on the target domain;
(BASELINE z) model trained on data from all source domains;
(a) a DNN model trained on the gold standard for the target domain (a ceiling rather than a baseline);
(b) IBCC over the models from source domains, tested on the target domain;
(c) Like (b) but with BSC-seq (note that BSC includes a linear model);

Other things we can consider including later if there is space:

(d) an ensemble containing the DNN from target domain and DNNs from source domains
(do other domain models provide complementary information? is a model trained on
just a few labels still useful, does it add noise, or does it get ignored?).
(e) a shallow linear model trained on the target domain (does a simpler model work better with very
small data?)
(f) linear model added to the ensemble.

'''
import os
import sys

import numpy as np
import json

from baselines.dawid_and_skene import ibccvb
from evaluation.experiment import output_root_dir
from experiments.AAAI2020.base_models import run_base_models
from bayesian_combination import bayesian_combination
from experiments.AAAI2020.helpers import evaluate, Dataset, get_anno_matrix, get_root_dir

reload = True
rerun_aggregators = True
verbose = False

if len(sys.argv) > 1:
    uberdomain = sys.argv[1]
else:
    uberdomain = 'TEd'

datadir = os.path.join(get_root_dir(), 'data/famulus_%s' % uberdomain)

nclasses = 9 # four types means I and B tags for each type + 1 O tag gives 9 different tags or class labels

base_models = ['bilstm-crf', 'crf']#'crf',  'flair-pos']#, 'flair-ner'] # 'flair' -- not implemented yet, do this if we can't get lample to work


print('Using base models: ' + str(base_models))

#iterate through the types of span we want to predict
for classid in [0, 1, 2, 3]:

    resdir = os.path.join(output_root_dir, 'famulus_%s_task1_type%i_basemodels%s' % (uberdomain, classid, '--'.join(base_models)) )
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

        for classid2 in [0, 1, 2, 3]:
            dataset2 = Dataset(datadir, classid2)
            if classid2 == classid:
                dataset = dataset2

            basepreds, basetrpreds, baseres = run_base_models(dataset2, classid2, uberdomain, base_model_str, reload)
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

                new_key = base_model_str + '_' + str(classid2) + '_' + key

                preds[new_key] = basepreds[key]
                trpreds[new_key] = basetrpreds[key]

    # BASELINE X -- majority vote from labellers trained on other domains
    preds['agg_MV'] = []
    trpreds['agg_MV'] = []
    res['agg_MV'] = []

    allgold = []
    alldocstart = []

    for didx, tedomain in enumerate(dataset.domains):
        votes, _ = get_anno_matrix(classid, preds, didx)
        counts = np.zeros((votes.shape[0], 3))
        for c in range(3): # iterate through IOB
            counts[:, c] = np.sum(votes==c, axis=1)

        mv = np.argmax(counts, axis=1)
        preds['agg_MV'].append(mv.tolist())

        res_s = evaluate(mv, dataset.tegold[tedomain], dataset.tedocstart[tedomain], f1type='all')
        res['agg_MV'].append(res_s)
        print('   Spantype %i: %i base labellers, F1 score=%s for majority vote tested on %s' %
              (classid, votes.shape[1], str(np.around(res_s, 2)), tedomain))

        allgold.append(dataset.tegold[tedomain])
        alldocstart.append(dataset.tedocstart[tedomain])

    cross_f1 = evaluate(np.concatenate(preds['agg_MV']),
                        np.concatenate(allgold),
                        np.concatenate(alldocstart),
                        f1type='all')
    print('*** Spantype %i, F1 score=%s for MV (micro average over domains) ' % (classid, str(cross_f1)) )
    print('*** Spantype %i, F1 score=%s for MV (macro average over domains) ' % (classid, str(np.mean(res['agg_MV'], axis=0))) )


    # Combination methods: IBCC and BSC-seq. Run the aggregation methods separately for each test domain. ------------------
    max_iter = 100
    alpha0_factor = 0.1
    alpha0_diags = 0.1
    beta0_factor = 0.1

    if rerun_aggregators or 'agg_ibcc' not in res or not len(res['agg_ibcc']):

        preds['agg_ibcc'] = []
        res['agg_ibcc'] = []

        for didx, tedomain in enumerate(dataset.domains):
            N = len(dataset.tetext[tedomain]) # number of test points
            # First, put the base labellers into a table.
            annos, uniform_priors = get_anno_matrix(classid, preds, didx, include_all=False)
            K = annos.shape[1] # number of annotators

            # now run IBCC
            probs, _ = ibccvb(annos, 3, beta0_factor, alpha0_factor, alpha0_diags, max_iter=max_iter,
                              uniform_priors=uniform_priors, verbose=True)
            agg = np.argmax(probs, axis=1)
            aggibcc = agg
            preds['agg_ibcc'].append(agg.flatten().tolist())

            res_s = evaluate(agg, dataset.tegold[tedomain], dataset.tedocstart[tedomain], f1type='all')
            res['agg_ibcc'].append(res_s)

            print('   Spantype %i: F1 score=%s for IBCC, tested on %s' % (classid, str(np.around(res_s, 2)), tedomain))

            with open(predfile, 'w') as fh:
                json.dump(preds, fh)

            with open(resfile, 'w') as fh:
                json.dump(res, fh)

    else:
        for didx, tedomain in enumerate(dataset.domains):
            print('   Spantype %i: F1 score=%f for IBCC, tested on %s' % (classid, res['agg_ibcc'][didx], tedomain))

    cross_f1 = evaluate(np.concatenate(preds['agg_ibcc']),
                        np.concatenate(allgold),
                        np.concatenate(alldocstart),
                        f1type='all')
    print('*** Spantype %i, F1 score=%s for IBCC (micro average over domains) ' % (classid, str(cross_f1)) )
    print('*** Spantype %i, F1 score=%s for IBCC (macro average over domains) ' % (classid, str(np.mean(res['agg_ibcc'], axis=0))) )

    alpha0_factor = 0.1
    alpha0_diags = 10
    beta0_factor = 0.1

    if rerun_aggregators or 'agg_bsc-seq' not in res or not len(res['agg_bsc-seq']):

        preds['agg_bsc-seq'] = []
        res['agg_bsc-seq'] = []

        for didx, tedomain in enumerate(dataset.domains):
            N = len(dataset.tetext[tedomain]) # number of test points
            # First, put the base labellers into a table.
            annos, uniform_priors = get_anno_matrix(classid, preds, didx, include_all=False)
            K = annos.shape[1] # number of annotators

            # now run BSC-seq
            bsc_model = bayesian_combination.BC(L=3, K=K, max_iter=max_iter, outside_label=1,
                            alpha0_diags=alpha0_diags, alpha0_factor=alpha0_factor, beta0_factor=beta0_factor,
                            annotator_model='seq', tagging_scheme='IOB2', taggers=[], true_label_model='HMM',
                            discrete_feature_likelihoods=True, converge_workers_first=False)

            bsc_model.verbose = False
            bsc_model.max_internal_iters = max_iter
            # why does Beta put a lot of weight on going from 2 to 0? Too much trust in 1 labels?
            probs, agg, pseq = bsc_model.fit_predict(annos, dataset.tedocstart[tedomain], dataset.tetext[tedomain])
            preds['agg_bsc-seq'].append(agg.flatten().tolist())

            aggprob = np.argmax(probs, axis=1)

            res_s = evaluate(agg, dataset.tegold[tedomain], dataset.tedocstart[tedomain], f1type='all')
            res['agg_bsc-seq'].append(res_s)
            print('   Spantype %i: F1 score=%s for BSC-seq, tested on %s' % (classid, str(np.around(res_s, 2)), tedomain))

        # save all the new results
        with open(predfile, 'w') as fh:
            json.dump(preds, fh)

        with open(resfile, 'w') as fh:
            json.dump(res, fh)

    else:
        for didx, tedomain in enumerate(dataset.domains):
            print('   Spantype %i: F1 score=%f for BSC-seq, tested on %s' % (classid, res['agg_bsc-seq'][didx], tedomain))

    cross_f1 = evaluate(np.concatenate(preds['agg_bsc-seq']),
                        np.concatenate(allgold),
                        np.concatenate(alldocstart),
                        f1type='all')
    print('*** Spantype %i, F1 score=%s for BSC-seq  (micro average over domains) ' % (classid, str(cross_f1)) )
    print('*** Spantype %i, F1 score=%s for BSC-seq  (macro average over domains) ' % (classid, str(np.mean(res['agg_bsc-seq'], axis=0))) )
