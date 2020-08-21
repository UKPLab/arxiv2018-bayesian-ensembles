import os, json, numpy as np
import sys

from experiments.AAAI2020.helpers import get_root_dir, Dataset, evaluate

if len(sys.argv) > 1:
    taskid = int(sys.argv[1])
else:
    taskid = 1

if len(sys.argv) > 2:
    base_models = sys.argv[2].split(',')
else:
    base_models = ['bilstm-crf', 'crf'] # , 'flair-pos', 'flair-ner']

if len(sys.argv) > 3:
    uberdomain = sys.argv[3]
else:
    uberdomain = 'TEd'

datadir = os.path.join(get_root_dir(), 'data/famulus_%s' % uberdomain)


totals_tok = {}
totals_rel = {}

for classid in range(0,4):
    class_tok = {}
    class_rel = {}

    basemodels_str = '--'.join(base_models)

    if taskid == 0:
        resdir = os.path.join(get_root_dir(), 'output/famulus_%s_results_%s_spantype%i'
                              % (uberdomain, base_models[0], classid))
    else:
        resdir = os.path.join(get_root_dir(), 'output/famulus_%s_task%i_type%i_basemodels%s' # _crfprobs
                      % (uberdomain, taskid, classid, basemodels_str))

    # resfile = os.path.join(resdir, 'res.json') this one contains the macro F1 scores, but this leads to some dodgy
    # results in small cases with almost no instances of a particular class.
    # with open(resfile, 'r') as fh:
    #     res = json.load(fh)

    # Open the predictions and compute the micro F1 scores:
    predfile = os.path.join(resdir, 'preds.json')
    with open(predfile, 'r') as fh:
        preds = json.load(fh)

    dataset = Dataset(datadir, classid, verbose=False)

    allgold = []
    alldocstart = []

    for didx, tedomain in enumerate(dataset.domains):
        allgold.append(dataset.tegold[tedomain])
        alldocstart.append(dataset.tedocstart[tedomain])

    allgold = np.concatenate(allgold)
    alldocstart = np.concatenate(alldocstart)

    res = {}
    for key in preds:

        # we only show results of the aggregation methods, not base models
        if (taskid > 0 and 'agg_' not in key) or len(preds[key]) == 0:
            continue

        cross_f1 = evaluate(np.concatenate(preds[key]),
                            allgold,
                            alldocstart,
                            f1type='all')


        # f1tok = 100*np.mean(res[key], 0)[1]
        # f1rel = 100*np.mean(res[key], 0)[3]
        f1tok = 100 * cross_f1[1]
        f1rel = 100 * cross_f1[3]

        print('Spantype %i: token f1 = %.1f; relaxed span f1 = %.1f, %s'
          % (classid, f1tok, f1rel, key))
        if key not in totals_tok:
            totals_tok[key] = f1tok
            totals_rel[key] = f1rel
        else:
            totals_tok[key] += f1tok
            totals_rel[key] += f1rel

        if key not in class_tok:
            class_tok[key] = f1tok
            class_rel[key] = f1rel
        else:
            class_tok[key] += f1tok
            class_rel[key] += f1rel

    totals_tok_cases = 0
    totals_rel_cases = 0
    for key in class_tok:
        if taskid == 0 and 'Case' in key:
            totals_tok_cases += class_tok[key]
            totals_rel_cases += class_rel[key]

    print('Spantype %i over all cases: %.1f, %.1f' % (classid, totals_tok_cases / 8, totals_rel_cases / 8))

totals_tok_cases = 0
totals_rel_cases = 0

for key in totals_tok:
    print('Mean: token f1 = %.1f; relaxed span f1 = %.1f, %s' %
          (totals_tok[key] / 4.0, totals_rel[key] / 4.0, key))

    if taskid == 0 and 'Case' in key:
        totals_tok_cases += totals_tok[key] / 4.0
        totals_rel_cases += totals_rel[key] / 4.0

print('Mean over all cases: %.1f, %.1f' % (totals_tok_cases / 8, totals_rel_cases / 8))