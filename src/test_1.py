'''

Hypothesis 1: ensemble learning with BCC improves performance with smaller amounts of data.


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

TODO: plot the results from the semi-supe bit.

TODO: Currently, the ensemble is weaker than the best base model (baseline_z). If baseline_z is excluded, MV is the best
aggregation method, likely because the models are too weak and are all of similar strength. When baseline_z is included,
the other models are not really adding independent errors except for noise.
We can try adding in more ensemble members:
- combine CRFs with the NNs; then there will be two baseline_z models, which may boost their chances
of being favoured by the aggregator
- NER
- POS tagger
- argument component tagger
Get these from flair! https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md
See terman models.

TODO: try the semi-supe setup with AL instead of random samples. With small labelled data size and
all labellers being quite weak, there is a good chance of biasing the weights in the wrong way. This
would affect BSC most, since the conf. matrix is very large.

TODO: Discuss with JP which f1 scores to use. The span-level ones may favour BSC. Also ask about overlapping annotations --
are they in a different dataset?

TODO: why does BSC produce strange results? Currently, the B token is becoming too popular, so that O->B is a popular
transition, even when there are very few B labels. We may need to step through to find out why.

'''
import os

import numpy as np
import json

from baselines.dawid_and_skene import ibccvb
from base_models import run_base_models
from bsc import bsc
from helpers import evaluate, Dataset, get_anno_matrix

reload = False
rerun_aggregators = True
verbose = False

datadir = os.path.expanduser('~/data/bayesian_sequence_combination/data/famulus_TEd')

nclasses = 9 # four types means I and B tags for each type + 1 O tag gives 9 different tags or class labels

base_models = ['bilstm-crf', 'crf']  # 'flair-pos', 'flair-ner'] # 'flair' -- not implemented yet, do this if we can't get lample to work

#iterate through the types of span we want to predict
for classid in [0, 1, 2, 3]:

    resdir = os.path.expanduser('~/data/bayesian_sequence_combination/output/famulus_TEd_task1_type%i' % classid)
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

            basepreds, basetrpreds, baseres = run_base_models(dataset2, classid2, base_model_str, reload)
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
    print('*** Cross-domain F1 score for majority vote: ' + str(cross_f1))
    print('*** Average F1 score for majority vote: ' + str(np.mean(res['agg_MV'], axis=0)) )

    # with open(predfile, 'w') as fh:
    #     json.dump(preds, fh)
    #
    # with open(trpredfile, 'w') as fh:
    #     json.dump(trpreds, fh)
    #
    # with open(resfile, 'w') as fh:
    #     json.dump(res, fh)

    # Combination methods: IBCC and BSC-seq. Run the aggregation methods separately for each test domain. ------------------
    max_iter = 100
    alpha0_factor = 0.1
    alpha0_diags = 0.1
    nu0_factor = 0.1

    if rerun_aggregators or 'agg_ibcc' not in res or not len(res['agg_ibcc']):

        preds['agg_ibcc'] = []
        res['agg_ibcc'] = []

        for didx, tedomain in enumerate(dataset.domains):
            N = len(dataset.tetext[tedomain]) # number of test points
            # First, put the base labellers into a table.
            annos, uniform_priors = get_anno_matrix(classid, preds, didx, include_all=False)
            K = annos.shape[1] # number of annotators

            # now run IBCC
            probs, _ = ibccvb(annos, 3, nu0_factor, alpha0_factor, alpha0_diags, max_iter, None, uniform_priors,
                              verbose=True)
            agg = np.argmax(probs, axis=1)
            aggibcc = agg
            preds['agg_ibcc'].append(agg.flatten().tolist())

            res_s = evaluate(agg, dataset.tegold[tedomain], dataset.tedocstart[tedomain], f1type='all')
            res['agg_ibcc'].append(res_s)

            print('   Spantype %i: F1 score=%s for IBCC, tested on %s' % (classid, str(np.around(res_s, 2)), tedomain))
    else:
        for didx, tedomain in enumerate(dataset.domains):
            print('   Spantype %i: F1 score=%f for IBCC, tested on %s' % (classid, res['agg_ibcc'][didx], tedomain))

    cross_f1 = evaluate(np.concatenate(preds['agg_ibcc']),
                        np.concatenate(allgold),
                        np.concatenate(alldocstart),
                        f1type='all')
    print('*** Cross-domain F1 score=%s for IBCC' % str(cross_f1) )
    print('*** Average F1 score=%s for IBCC' % str(np.mean(res['agg_ibcc'], axis=0)) )


    if rerun_aggregators or 'agg_bsc-seq' not in res or not len(res['agg_bsc-seq']):

        preds['agg_bsc-seq'] = []
        res['agg_bsc-seq'] = []

        for didx, tedomain in enumerate(dataset.domains):
            N = len(dataset.tetext[tedomain]) # number of test points
            # First, put the base labellers into a table.
            annos, uniform_priors = get_anno_matrix(classid, preds, didx, include_all=False)
            K = annos.shape[1] # number of annotators

            # now run BSC-seq
            bsc_model = bsc.BSC(L=3, K=K, max_iter=max_iter, before_doc_idx=1,
                        alpha0_diags=alpha0_diags, alpha0_factor=alpha0_factor, beta0_factor=nu0_factor,
                        worker_model='seq', tagging_scheme='IOB2', data_model=[], transition_model='HMM',
                        no_words=False)

            bsc_model.verbose = False
            bsc_model.max_internal_iters = max_iter
            # why does Beta put a lot of weight on going from 2 to 0? Too much trust in 1 labels?
            probs, agg, pseq = bsc_model.run(annos, dataset.tedocstart[tedomain], dataset.tetext[tedomain],
                                             converge_workers_first=False, uniform_priors=uniform_priors)
            preds['agg_bsc-seq'].append(agg.flatten().tolist())

            aggprob = np.argmax(probs, axis=1)

            res_s = evaluate(agg, dataset.tegold[tedomain], dataset.tedocstart[tedomain], f1type='all')
            res['agg_bsc-seq'].append(res_s)
            print('   Spantype %i: F1 score=%s for BSC-seq, tested on %s' % (classid, str(np.around(res_s, 2)), tedomain))

        # save all the new results
        with open(predfile, 'w') as fh:
            json.dump(preds, fh)

        with open(trpredfile, 'w') as fh:
            json.dump(trpreds, fh)

        with open(resfile, 'w') as fh:
            json.dump(res, fh)

    else:
        for didx, tedomain in enumerate(dataset.domains):
            print('   Spantype %i: F1 score=%f for BSC-seq, tested on %s' % (classid, res['agg_bsc-seq'][didx], tedomain))

    cross_f1 = evaluate(np.concatenate(preds['agg_bsc-seq']),
                        np.concatenate(allgold),
                        np.concatenate(alldocstart),
                        f1type='all')
    print('*** Cross-domain F1 score=%s for BSC-seq' % str(cross_f1) )
    print('*** Average F1 score=%s for BSC-seq' % str(np.mean(res['agg_bsc-seq'], axis=0)) )

    # SEMI-SUPERVISED: use a little bit of training data from the target domain --------------------------------------------
    #
    # preds['agg_ibcc_SEMISUPE'] = []
    # preds['agg_bsc-seq_SEMISUPE'] = []
    #
    # res['agg_ibcc_SEMISUPE'] = []
    # res['agg_bsc-seq_SEMISUPE'] = []
    #
    # # Now we do a simulation with different amounts of data from the target domain
    # for didx, tedomain in enumerate(domains):
    #
    #     N = len(tetext[tedomain]) # number of test points
    #     # First, put the base labellers into a table.
    #     annos, uniform_priors = get_anno_matrix(tedomain, preds)
    #     K = annos.shape[1]  # number of annotators
    #
    #     batchsize = 10
    #     nbatches = 6
    #
    #     preds['agg_ibcc_SEMISUPE'].append([])
    #     res['agg_ibcc_SEMISUPE'].append([])
    #
    #     preds['agg_bsc-seq_SEMISUPE'].append([])
    #     res['agg_bsc-seq_SEMISUPE'].append([])
    #
    #     # take the training labels for aggregation methods from the target domain's training set.
    #     # -- this means that the test set for the target domain stays the same.
    #     Ntargettr = len(trdocstart[tedomain])
    #     Ndoc = np.sum(trdocstart[tedomain])
    #     docids = np.cumsum(trdocstart[tedomain])
    #
    #     print('Domain %s has %i training docs and %i training tokens' % (tedomain, Ndoc, Ntargettr))
    #
    #     # get the annotations for the training set
    #     trannos, _ = get_anno_matrix(tedomain, trpreds)
    #
    #     #annos_b = np.concatenate((annos, trannos), axis=0)
    #
    #     for b in range(1, nbatches):
    #         # get the training labels for the training set
    #         trdocs = np.random.choice(Ndoc, b*batchsize, replace=False)
    #         tridxs = np.in1d(docids, trdocs)
    #         #trlabels = np.zeros(Ntargettr) - 1
    #         #trlabels[tridxs] = trgold[tedomain][tridxs]
    #         trlabels = trgold[tedomain][tridxs]
    #         trlabels = np.concatenate((np.zeros(N)-1, trlabels)) # concatenate with a vector of missing training labels for the test items
    #         annos_b = np.concatenate((annos, trannos[tridxs]), axis=0)
    #
    #         # now run IBCC
    #         probs, _ = ibccvb(annos_b, nclasses, nu0_factor, alpha0_factor, alpha0_diags, max_iter, trlabels)
    #         agg = np.argmax(probs, axis=1)
    #         agg = agg[:N] # ignore the training items
    #         preds['agg_ibcc_SEMISUPE'][didx].append(agg.flatten().tolist())
    #
    #         res_s = evaluate(agg, tegold[tedomain], tedocstart[tedomain])
    #         res['agg_ibcc_SEMISUPE'][didx].append(res_s)
    #         print('F1 score=%s for IBCC with %i training docs on %s' % (str(res_s), b*batchsize, tedomain))
    #
    #         # now run BSC-seq
    #         bsc_model = bsc.BSC(L=nclasses, K=K, max_iter=max_iter, before_doc_idx=1, inside_labels=inside_labels,
    #                         outside_labels=outside_labels, beginning_labels=begin_labels,
    #                         alpha0_diags=alpha0_diags, alpha0_factor=alpha0_factor, beta0_factor=nu0_factor,
    #                         worker_model='seq', tagging_scheme='IOB2', data_model=[], transition_model='HMM', no_words=False)
    #
    #         bsc_model.verbose = False
    #         bsc_model.max_internal_iters = max_iter
    #
    #         docstart = np.concatenate((tedocstart[tedomain], trdocstart[tedomain][tridxs]))
    #         text     = np.concatenate((tetext[tedomain], trtext[tedomain][tridxs]))
    #
    #         probs, agg, pseq = bsc_model.run(annos_b, docstart, text, converge_workers_first=False, gold_labels=trlabels)
    #         agg = agg[:N] # forget the training points
    #         preds['agg_bsc-seq_SEMISUPE'][didx].append(agg.flatten().tolist())
    #
    #         res_s = evaluate(agg, tegold[tedomain], tedocstart[tedomain])
    #         res['agg_bsc-seq_SEMISUPE'][didx].append(res_s)
    #         print('F1 score=%s for BSC-seq with %i training docs on %s' % (str(res_s), b*batchsize, tedomain))
    #
    # print('Average F1 scores for IBCC with increasing amounts of training data:')
    # print(np.mean(np.array(res['agg_ibcc_SEMISUPE']), axis=(0,1)).tolist())
    #
    # print('Average F1 scores for BSC-seq with increasing amounts of training data:')
    # print(np.mean(np.array(res['agg_bsc-seq_SEMISUPE']), axis=(0,1)).tolist())
    #
    # # save all the new results
    # with open(predfile, 'w') as fh:
    #     json.dump(preds, fh)
    #
    # with open(trpredfile, 'w') as fh:
    #     json.dump(trpreds, fh)
    #
    # with open(resfile, 'w') as fh:
    #     json.dump(res, fh)

    reload = True # reload the base classifiers even if we retrained them in the first iteration