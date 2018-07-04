'''
Created on April 27, 2018

@author: Edwin Simpson


TODO currently checking why the early stopping was not working in BAC integrateLSTM


'''
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_annotator_combination/output/ner-by-sentence/'

regen_data = True
gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val = \
    load_data.load_ner_data(regen_data)

# debug with subset -------
# s = 100
# idxs = np.argwhere(gt!=-1)[:s, 0]
#
# idxs = (gt != -1).flatten()

# gt = gt[idxs]
# annos = annos[idxs]
# doc_start = doc_start[idxs]
# print('No. documents:')
# print(np.sum(doc_start))
# text = text[idxs]
# gt_task1_val = gt_task1_val[idxs]
#
# ntest = 100
# doc_start_nocrowd = doc_start_nocrowd[:ntest]
# text_nocrowd = text_nocrowd[:ntest]
# gt_nocrowd = gt_nocrowd[:ntest]
#
# -------------------------

exp = Experiment(None, 9, annos.shape[1], None, max_iter=20)
exp.save_results = True
exp.opt_hyper = False#True

# exp.nu0_factor = 0.001
# exp.alpha0_diags = 0.8
# exp.alpha0_factor = 0.1

exp.nu0_factor = 0.1
exp.alpha0_acc_bias = 0

exp.alpha0_diags = 10
exp.alpha0_factor = 1
best_bac_wm = 'bac_seq'

# exp.alpha0_diags = 10
# exp.alpha0_factor = 0.1
# best_bac_wm = 'bac_vec'

# exp.alpha0_diags = 10
# exp.alpha0_factor = 0.1
# best_bac_wm = 'bac_mace'

# exp.alpha0_diags = 1
# exp.alpha0_factor = 1
# best_bac_wm = 'bac_acc'

# exp.alpha0_diags = 0.1
# exp.alpha0_factor = 0.1
# best_bac_wm = 'bac_ibcc'

# run all the methods that don't require tuning here
exp.methods =  [
                #'majority',
                # 'ds',
                #'gt_then_LSTM',
                #best_bac_wm
                best_bac_wm + '_integrateBOF',
                best_bac_wm + '_integrateBOF_integrateLSTM_atEnd',
                best_bac_wm + '_integrateBOF_then_LSTM',
                best_bac_wm + '_integrateBOF_integrateLSTM',
                #best_bac_wm + '_integrateBOF_noHMM',
                #best_bac_wm + '_noHMM'
]

# should run both task 1 and 2.
exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    new_data=regen_data
)
