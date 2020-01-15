'''
Created on April 27, 2018

@author: Edwin Simpson
'''
import os

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

# currently on wormulon:
# testing with priors 1, 10, 10, 1
# testing BSC-cv.

# Now, in this script: try reloading the older data files. It is possible they were produced from the random sample?
# -- probably seed 10 for 160818 or 170518 version, probably seed 2348945 for the 170518 or older missing versions.
# This should not depend on the loading order of the corpus on a particular machine -- the docs are loaded in the
# order they are first mentioned in acl17-test/PICO-annos-crowdsourcing.json.
# The bio_apu folder might actually have been generated recently when I tested all the steps.
# So up until then, it may have been using 160818 or 1705118.

# Longer-term question -- why does it make so much difference to BSC but not to HMM-crowd? What is it sensitive to?
# Since performance on Lichtenberg is also a bit different, is there some kind of machine precision error that we come
# up against? Or perhaps HMMcrowd hits the precision error earlier?
regen_data = False

# datadir = 'bio_160818maybe'
# gt, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev = \
#     load_data.load_biomedical_data(regen_data, data_folder=datadir)#, debug_subset_size=50000)
# #
# # this is the one we used in the paper, result_started-2019-08-22-06-17-54-Nseen56858.csv
# best_nu0factor = 1
# best_diags = 100
# best_factor = 10
# best_outside_factor = 5
#
# # ------------------------------------------------------------------------------------------------
# exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, outside_factor=best_outside_factor)
#
# exp.save_results = True
# exp.opt_hyper = False
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
#
# # # run all the methods that don't require tuning here
# exp.methods =  [
#                 'bac_seq_integrateIF',
# ]
#
# output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon_%f_%f_%f_%f_%s_betaOO'
#                           % (best_nu0factor, best_diags, best_factor, best_outside_factor, datadir))
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                  ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                  new_data=regen_data
#                  )

#-------------------------------------------------------------------------------------------------
# datadir = 'bio_170518'
# gt, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev = \
#     load_data.load_biomedical_data(regen_data, data_folder=datadir)#, debug_subset_size=50000)
# #
# # this is the one we used in the paper, result_started-2019-08-22-06-17-54-Nseen56858.csv
# best_nu0factor = 1
# best_diags = 10
# best_factor = 100
# best_outside_factor = 5
#
# # ------------------------------------------------------------------------------------------------
# exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, outside_factor=best_outside_factor)
#
# exp.save_results = True
# exp.opt_hyper = False
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
#
# # # run all the methods that don't require tuning here
# exp.methods =  [
#                 'bac_seq_integrateIF',
# ]
#
# output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon_%f_%f_%f_%f_%s'
#                           % (best_nu0factor, best_diags, best_factor, best_outside_factor, datadir))
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                  ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                  new_data=regen_data
#                  )
#-------------------------------------------------------------------------------------------------

# now testing: what happens if we remove the outside factor stuff? -- we get higher recall but prec is a little too low
# + does HMM crowd still perform well with the vb[1] = 1?
# Now: outside factor of 5 was better than 1. How about 10? What if we increase the diags relative to the factors still?
# Trying next: should the beta fix be changed back to the more sensible setting to increase recall?
# Next to-do: should both vb[1] and nu0 in BSC be set back to 0.1? Or a higher value?
# Other minor tweak we could test: before_idx?
# Another idea: reduce the prior for beta on B-O transitions to encourage long spans. E.g. add disallowed count from O->I to B->I instead of O->O
# For BSC-CV: need to decrease precision slightly and increase recall. Can do by using nu0_words=0.1? For BSC-seq this harmed prec a little much. Try also the alterative setting commented out in the code?
datadir = 'bio'
gt, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev = \
    load_data.load_biomedical_data(regen_data, data_folder=datadir) # , debug_subset_size=50000)

# # ------------------------------------------------------------------------------------------------
# # Almost. Lower outside factor to somewhere between 1 and 2 to boost recall and drop prec slightly
# best_nu0factor = 1
# best_diags = 100
# best_factor = 10
# best_outside_factor = 1.3
#
# exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, outside_factor=best_outside_factor)
#
# exp.save_results = True
# exp.opt_hyper = False
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
#
# # # run all the methods that don't require tuning here
# exp.methods =  [
#                 'bac_seq_integrateIF',
# ]
#
# output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon_%f_%f_%f_%f_%s_betaOO_nu0_1'
#                           % (best_nu0factor, best_diags, best_factor, best_outside_factor, datadir))
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                  ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                  new_data=regen_data
#                  )

# ------------------------------------------------------------------------------------------------
# # THE BEST SO FAR
# best_nu0factor = 1
# best_diags = 10
# best_factor = 10
# best_outside_factor = 10
#
# exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, begin_factor=best_outside_factor)
#
# exp.save_results = True
# exp.opt_hyper = False
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
#
# # # run all the methods that don't require tuning here
# exp.methods =  [
#                 'ibcc',
#                 'bac_seq_integrateIF',
# ]
#
# output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon_%f_%f_%f_%f_%s_betaOO_nu0_1_prior'
#                           % (best_nu0factor, best_diags, best_factor, best_outside_factor, datadir))
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                  ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                  new_data=regen_data
#                  )

# THE BEST SO FAR, but with different outside factor
best_nu0factor = 1
best_diags = 10
best_factor = 10
best_outside_factor = 1

exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, begin_factor=best_outside_factor)

exp.save_results = True
exp.opt_hyper = False

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor

# # run all the methods that don't require tuning here
exp.methods =  [
                'ibcc',
                'bac_seq_integrateIF',
]

output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon_%f_%f_%f_%f_%s_betaOO_nu0_1_prior'
                          % (best_nu0factor, best_diags, best_factor, best_outside_factor, datadir))

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text,
                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                 new_data=regen_data
                 )
# THE BEST SO FAR
best_nu0factor = 1
best_diags = 10
best_factor = 10
best_outside_factor = 2

exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, begin_factor=best_outside_factor)

exp.save_results = True
exp.opt_hyper = False

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor

# # run all the methods that don't require tuning here
exp.methods =  [
                'ibcc',
                'bac_seq_integrateIF',
]

output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon_%f_%f_%f_%f_%s_betaOO_nu0_1_prior'
                          % (best_nu0factor, best_diags, best_factor, best_outside_factor, datadir))

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text,
                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                 new_data=regen_data
                 )
# THE BEST SO FAR
best_nu0factor = 1
best_diags = 10
best_factor = 10
best_outside_factor = 5

exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, begin_factor=best_outside_factor)

exp.save_results = True
exp.opt_hyper = False

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor

# # run all the methods that don't require tuning here
exp.methods =  [
                'ibcc',
                'bac_seq_integrateIF',
]

output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon_%f_%f_%f_%f_%s_betaOO_nu0_1_prior'
                          % (best_nu0factor, best_diags, best_factor, best_outside_factor, datadir))

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text,
                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                 new_data=regen_data
                 )

# # from early paper draft
# best_nu0factor = 100
# best_diags = 1
# best_factor = 1
# best_outside_factor = 1
#
# exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, begin_factor=best_outside_factor)
#
# exp.save_results = True
# exp.opt_hyper = False
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
#
# # # run all the methods that don't require tuning here
# exp.methods =  [
#                 'bac_seq_integrateIF',
# ]
#
# output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon_%f_%f_%f_%f_%s_betaOO_nu0_1_prior'
#                           % (best_nu0factor, best_diags, best_factor, best_outside_factor, datadir))
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                  ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                  new_data=regen_data
#                  )
#
# # this is the one we used in the paper, result_started-2019-08-22-06-17-54-Nseen56858.csv
# best_nu0factor = 1
# best_diags = 10
# best_factor = 100
# best_outside_factor = 10
#
# exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, begin_factor=best_outside_factor)
#
# exp.save_results = True
# exp.opt_hyper = False
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
#
# # # run all the methods that don't require tuning here
# exp.methods =  [
#                 'bac_seq_integrateIF',
# ]
#
# output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon_%f_%f_%f_%f_%s_betaOO_nu0_1_prior'
#                           % (best_nu0factor, best_diags, best_factor, best_outside_factor, datadir))
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                  ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                  new_data=regen_data
#                  )

# # ------------------------------------------------------------------------------------------------
# # new best-so-far but has slightly low prec and high rec
# best_nu0factor = 1
# best_diags = 100
# best_factor = 100
# best_outside_factor = 1
#
# exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, begin_factor=best_outside_factor)
#
# exp.save_results = True
# exp.opt_hyper = False
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
#
# # # run all the methods that don't require tuning here
# exp.methods =  [
#                 'bac_seq_integrateIF',
# ]
#
# output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon_%f_%f_%f_%f_%s_betaOO_nu0_1_prior'
#                           % (best_nu0factor, best_diags, best_factor, best_outside_factor, datadir))
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                  ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                  new_data=regen_data
#                  )

#-----------------------------------------------------------------------------------------------
# datadir = 'bio_160818maybe'
# gt, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev = \
#     load_data.load_biomedical_data(regen_data, data_folder=datadir)#, debug_subset_size=50000)
#
# ------------------------------------------------------------------------------------------------
# this is the one we used in the paper, result_started-2019-08-22-06-17-54-Nseen56858.csv
# best_nu0factor = 0.1
# best_diags = 0
# best_factor = 0.1
# best_outside_factor = 1
#
# exp = Experiment(None, 3, annos.shape[1], None, max_iter=20, begin_factor=best_outside_factor)
#
# exp.save_results = True
# exp.opt_hyper = False
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
#
# # # run all the methods that don't require tuning here
# exp.methods =  [
#                 'HMM_crowd',
# ]
#
# output_dir = os.path.join(load_data.output_root_dir, 'pico_wormulon_%f_%f_%f_%s_alpha0factortwice'
#                           % (best_nu0factor, best_diags, best_factor, datadir))
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                  ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                  new_data=regen_data
#                  )

# # ------------------------------------------------------------------------------------------------
# nu_factors = [0.01, 0.1, 1]
# diags = [0.1, 1, 10]
# factors = [0.1, 1, 10]
#
# methods_to_tune = [
#                    'bac_seq_integrateIF',
#                    ]
#
# best_bac_wm_score = -np.inf
#
# # tune with small dataset to save time
# idxs = np.argwhere(gt_task1_dev != -1)[:, 0]
# idxs = np.concatenate((idxs, np.argwhere(gt != -1)[:, 0]))
# idxs = np.concatenate((idxs, np.argwhere((gt == -1) & (gt_task1_dev == -1))[:300, 0]))  # 100 more to make sure the dataset is reasonable size
#
# tune_gt_dev = gt_task1_dev[idxs]
# tune_annos = annos[idxs]
# tune_doc_start = doc_start[idxs]
# tune_text = text[idxs]

#for m, method in enumerate(methods_to_tune):
#    print('TUNING %s' % method)
#
#    best_scores = exp.tune_alpha0(diags, factors, nu_factors, method, tune_annos, tune_gt_dev, tune_doc_start,
#                                  output_dir, tune_text, metric_idx_to_optimise=11)
#    best_idxs = best_scores[1:].astype(int)
#    exp.nu0_factor = nu_factors[best_idxs[0]]
#    exp.alpha0_diags = diags[best_idxs[1]]
#    exp.alpha0_factor = factors[best_idxs[2]]
#
#    print('Best values: %f, %f, %f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
#
#    # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
#    exp.methods = [method]
#    exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, return_model=True,
#                ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                new_data=regen_data
#    )

# # run all the methods that don't require tuning here
# exp.methods =  [
#                 'bac_seq_integrateIF_then_LSTM',
#                 'bac_seq_integrateIF_integrateLSTM_atEnd',
# ]
#
# # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
# exp.run_methods(annos, gt, doc_start, output_dir, text,
#                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
#                 new_data=regen_data
#                 )
