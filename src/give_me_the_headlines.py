import os, json, numpy as np

from helpers import get_root_dir

base_models = ['bilstm-crf', 'crf'] # , 'flair-pos', 'flair-ner']

taskid = 1

for classid in range(4):
    basemodels_str = '--'.join(base_models)

    resdir = os.path.join(get_root_dir(), 'output/famulus_TEd_task%i_type%i_basemodels%s'
                      % (taskid, classid, basemodels_str))

    resfile = os.path.join(resdir, 'res.json')

    with open(resfile, 'r') as fh:
        res = json.load(fh)

    print('Spantype %i: token f1 = %f; relaxed span f1 = %f'
          % (classid, np.mean(res,0)[1], np.mean(res,0)[3]))
