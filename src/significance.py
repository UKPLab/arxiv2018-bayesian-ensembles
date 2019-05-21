import json, numpy as np, pandas as pd, os
from scipy.stats import wilcoxon

outroot = os.path.expanduser('~/data/bayesian_sequence_combination/output/')

baseline = outroot + '/ner/pred_started-2019-05-17-09-52-22-Nseen29704.csv'

bscseq = outroot + '/ner/pred_started-2019-04-11-10-24-38-Nseen29704.csv'

preds_base = pd.read_csv(baseline)

preds_bscseq = pd.read_csv(bscseq)

_, p = wilcoxon(preds_base, preds_bscseq)

print('NER: BSC-seq %s significantly different with p = %f' % ('is' if p<0.05 else 'ISNT', p))

#####

baseline = outroot + '/pico2/pred_started-2019-05-11-14-24-25-Nseen56858.csv'

bscseq = outroot + '/bio_task1/pred_started-2018-08-25-18-08-57-Nseen56858.csv'

preds_base = pd.read_csv(baseline)

preds_bscseq = pd.read_csv(bscseq)

_, p = wilcoxon(preds_base, preds_bscseq)

print('PICO: BSC-seq %s significantly different with p = %f' % ('is' if p<0.05 else 'ISNT', p))

#####

baseline = outroot + '/arg_LMU_final/pred_started-2019-05-10-18-50-04-Nseen40000.csv'

bscseq = outroot + '/arg_LMU_final/pred_started-2019-05-09-23-33-51-Nseen40000.csv'

preds_base = pd.read_csv(baseline)

preds_bscseq = pd.read_csv(bscseq)

_, p = wilcoxon(preds_base, preds_bscseq)

print('ARG: BSC-seq %s significantly different with p = %f' % ('is' if p<0.05 else 'ISNT', p))