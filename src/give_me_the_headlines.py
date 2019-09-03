import os, json, numpy as np
import sys

from helpers import get_root_dir

if len(sys.argv) > 1:
    taskid = int(sys.argv[1])
else:
    taskid = 1

if len(sys.argv) > 2:
    base_models = sys.argv[2].split(',')
else:
    base_models = ['bilstm-crf', 'crf'] # , 'flair-pos', 'flair-ner']

totals_tok = {}
totals_rel = {}

for classid in range(4):
    basemodels_str = '--'.join(base_models)

    resdir = os.path.join(get_root_dir(), 'output/famulus_TEd_task%i_type%i_basemodels%s'
                      % (taskid, classid, basemodels_str))

    resfile = os.path.join(resdir, 'res.json')

    with open(resfile, 'r') as fh:
        res = json.load(fh)

    for key in res:
        f1tok = 100*np.mean(res[key], 0)[1]
        f1rel = 100*np.mean(res[key], 0)[3]

        print('Spantype %i: token f1 = %.1f; relaxed span f1 = %.1f, %s'
          % (classid, f1tok, f1rel, key))
        if key not in totals_tok:
            totals_tok[key] = f1tok
            totals_rel[key] = f1rel
        else:
            totals_tok[key] += f1tok
            totals_rel[key] += f1rel

for key in totals_tok:
    print('Mean: token f1 = %.1f; relaxed span f1 = %.1f, %s' %
          (totals_tok[key] / 4.0, totals_rel[key] / 4.0, key))