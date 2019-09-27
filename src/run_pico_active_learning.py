'''
Created on April 27, 2018

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np
import os

gt, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev = load_data.load_biomedical_data(False)

# run with small dataset to save time
idxs = np.argwhere(gt != -1)[:, 0]
idxs = np.concatenate((idxs, np.argwhere(gt != -1)[:, 0]))
idxs = np.concatenate((idxs, np.argwhere((gt == -1) & (gt_task1_dev == -1))[:300, 0]))  # 100 more to make sure the dataset is reasonable size

gt_task1_dev = gt_task1_dev[idxs]
annos = annos[idxs]
doc_start = doc_start[idxs]
text = text[idxs]
gt = gt[idxs]

gt_dev = gt_dev[:100]
doc_start_dev = doc_start_dev[:100]
text_dev = text_dev[:100]

num_reps = 10
for rep in range(1, num_reps):
    # Random Sampling ------------------------------------------------------------------------------

    # output_dir = '../../data/bayesian_sequence_combination/output/bio_rand_small/'
    # if not os.path.isdir(output_dir):
    #     os.mkdir(output_dir)
    #
    # exp = Experiment(None, 3, annos.shape[1], None, max_iter=10, crf_probs=False, rep=rep)
    #
    # exp.save_results = True
    # exp.opt_hyper = False #True
    #
    # exp.nu0_factor = 100
    # exp.alpha0_diags = 1
    # exp.alpha0_factor = 1
    #
    # exp.methods = [
    #     'bac_seq_integrateIF_integrateLSTM_atEnd',
    #                ]
    #
    # exp.save_results = True
    # exp.opt_hyper = False #True
    #    exp.random_sampling = True

    # exp.run_methods(annos, gt, doc_start, output_dir, text,
    #                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
    #                 active_learning=True, AL_batch_fraction=1.0)


    output_dir = os.path.join(load_data.output_root_dir, 'bio_al_small')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    exp = Experiment(None, 3, annos.shape[1], None, max_iter=10, crf_probs=True, rep=rep)

    exp.save_results = True
    exp.opt_hyper = False #True

    exp.nu0_factor = 100
    exp.alpha0_diags = 1
    exp.alpha0_factor = 1

    exp.methods = [
        'bac_seq_integrateIF_integrateLSTM_atEnd'
                   ]

    exp.save_results = True
    exp.opt_hyper = False #True

    exp.run_methods(annos, gt, doc_start, output_dir, text,
                    ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                    active_learning=True, AL_batch_fraction=1)

    # ------------------------

    output_dir = os.path.join(load_data.output_root_dir, 'bio_al_small')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    exp = Experiment(None, 3, annos.shape[1], None, max_iter=10, crf_probs=True, rep=rep)

    exp.save_results = True
    exp.opt_hyper = False  # True

    exp.nu0_factor = 100
    exp.alpha0_diags = 1
    exp.alpha0_factor = 1

    exp.methods = [
        'bac_seq_integrateIF_then_LSTM',
        #'HMM_crowd_then_LSTM'
    ]

    exp.save_results = True
    exp.opt_hyper = False  # True

    exp.run_methods(annos, gt, doc_start, output_dir, text,
                    ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                    active_learning=True, AL_batch_fraction=1)

    # Random Sampling ------------------------------------------------------------------------------

    # output_dir = os.path.join(load_data.output_root_dir, 'bio_rand_small')
    # if not os.path.isdir(output_dir):
    #     os.mkdir(output_dir)
    #
    # exp = Experiment(None, 3, annos.shape[1], None, max_iter=10, crf_probs=False, rep=rep)
    #
    # exp.save_results = True
    # exp.opt_hyper = False  # True
    #
    # exp.nu0_factor = 100
    # exp.alpha0_diags = 1
    # exp.alpha0_factor = 1
    #
    # exp.methods = [
    #     'bac_seq_integrateIF_then_LSTM',
    #     'HMM_crowd_then_LSTM'
    # ]
    #
    # exp.save_results = True
    # exp.opt_hyper = False  # True
    #    exp.random_sampling = True

    # exp.run_methods(annos, gt, doc_start, output_dir, text,
    #                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
    #                 active_learning=True, AL_batch_fraction=1.0)