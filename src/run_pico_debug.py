'''
Created on April 27, 2018

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_sequence_combination/output/pico-debug/'

regen_data = False

gt, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev = \
   load_data.load_biomedical_data(regen_data, debug_subset_size=50000)

# gt = np.array([2, 0, 0, 0, 1])
# annos = np.array([[2, 2, 2],
#                   [0, 0, 0],
#                   [0, 0, 0],
#                   [0, 0, 0],
#                   [1, 1, 1]])
# doc_start = np.array([1, 0, 0, 0, 0])[:, None]
# text = np.array(['do', 'be', 'do', 'wap', 'blach'])

gt_dev = None
doc_start_dev = None
text_dev = None

np.savetxt('./data/pico-debug-gt.csv', gt)
np.savetxt('./data/pico-debug-annos.csv', annos)


exp = Experiment(None, 3, annos.shape[1], None, max_iter=20)

exp.save_results = True
exp.opt_hyper = False #True

exp.alpha0_diags = 100
exp.alpha0_factor = 9

best_bac_wm = 'bac_seq' # choose model with best score for the different BAC worker models
best_nu0factor = 0.1#100
best_diags = 10
best_factor = 1

nu_factors = [0.1, 10, 100]
diags = [0.1, 1, 10, 100] #, 50, 100]#[1, 50, 100]#[1, 5, 10, 50]
factors = [0.1, 1, 10]

best_bac_wm_score = -np.inf

print('best BAC method = %s' % best_bac_wm)

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor

# # exp.nu0_factor = .1
# # exp.alpha0_diags = .1
# # exp.alpha0_factor = .1
#
# # run all the methods that don't require tuning here
exp.methods =  [
    'bac_seq_integrateIF',
    #'bac_ibcc_integrateIF',
    #'majority'
]

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text,
                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                new_data=regen_data
                )

