import os
import sys

import evaluation.experiment
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

if __name__ == '__main__':

    # Process script arguments --------------------------------------------------------------
    if len(sys.argv) > 1:
        second_batch_workers_only = bool(int(sys.argv[1]))
    else:
        second_batch_workers_only = False

    if len(sys.argv) > 2:
        gold_labelled_only = bool(int(sys.argv[2]))
    else:
        gold_labelled_only = False

    if len(sys.argv) > 3:
        regen_data = bool(int(sys.argv[3]))
    else:
        regen_data = False

    print('Running ' + ('with' if second_batch_workers_only else 'without') + ' second-batch workers only.')

    # Load ARG dataset ---------------------------------------------------------------------

    N = 0 # set to 0 to use all, any other number will debug on a subset
    gt, annos, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev = load_data.load_arg_sentences(
        N, regen_data, second_batch_workers_only, gold_labelled_only)
    N = float(len(gt))

    valid_workers = np.any(annos != -1, axis=0)
    print('Valid workers for this subset are %s' % str(np.argwhere(valid_workers).flatten()))

    nclasses = 5

    # Run with hyperparameter tuning -------------------------------------------------------

    # TUNE on F1 score.
    # If multiple hyperparameters produce same F1, we could use CEE as a tie-breaker?
    output_dir = os.path.join(evaluation.experiment.output_root_dir, 'arg_LMU_strictF1')
    beta_factors = [0.1, 1, 10]
    diags = [0.1, 1, 10]
    factors = [0.1, 1, 10, 100]

    exp = Experiment(output_dir, nclasses, annos, gt, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev,
                     max_iter=20, begin_factor=5.0)

    methods_to_tune = [
        'bac_seq_integrateIF',
    ]

    for m, method in enumerate(methods_to_tune):
        print('TUNING %s' % method)

        best_scores = exp.tune_alpha0(diags, factors, beta_factors, method, new_data=regen_data)
        best_idxs = best_scores[1:].astype(int)
        exp.beta0_factor = beta_factors[best_idxs[0]]
        exp.alpha0_diags = diags[best_idxs[1]]
        exp.alpha0_factor = factors[best_idxs[2]]

        print('Best values for %s: %f, %f, %f' % (method, exp.beta0_factor, exp.alpha0_diags, exp.alpha0_factor))

        # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
        exp.methods = [method]
        exp.run_methods(new_data=regen_data)

    # -------------------------------------------------------------------------
    exp = Experiment(output_dir, nclasses, annos, gt, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev,
                     max_iter=20)

    methods_to_tune = [
        'ds',
        'ibcc',
        'bac_ibcc_integrateIF',
        'bac_mace_integrateIF',
        'bac_vec_integrateIF',
    ]

    for m, method in enumerate(methods_to_tune):
        print('TUNING %s' % method)

        best_scores = exp.tune_alpha0(diags, factors, beta_factors, method, new_data=regen_data)
        best_idxs = best_scores[1:].astype(int)
        exp.beta0_factor = beta_factors[best_idxs[0]]
        exp.alpha0_diags = diags[best_idxs[1]]
        exp.alpha0_factor = factors[best_idxs[2]]

        print('Best values for %s: %f, %f, %f' % (method, exp.beta0_factor, exp.alpha0_diags, exp.alpha0_factor))

        # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
        exp.methods = [method]
        exp.run_methods(new_data=regen_data)

    # Run with optimal values found during our experiments by tuning on dev set ---------------------------------
    # these methods do not take the tuned values; MACE defaults were best
    output_dir = os.path.join(evaluation.experiment.output_root_dir, 'arg_LMU')
    exp = Experiment(output_dir, nclasses, annos, gt, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev,
                     max_iter=20)
    exp.methods = [
        'majority',
        'best',
        'worst',
        'ds',
        'mace'
    ]
    exp.run_methods(new_data=regen_data)

    # -------------------------------------------------------------------------
    beta0_factor = 10
    alpha0_diags = 0.1
    alpha0_factor = 10
    # 1, 1, 10 gives similar results -- not optimal on dev, but was tried as default values before tuning...
    output_dir = os.path.join(evaluation.experiment.output_root_dir, 'arg_LMU_%f_%f_%f' % (beta0_factor, alpha0_diags, alpha0_factor))
    exp = Experiment(output_dir, nclasses, annos, gt, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev,
                     beta0_factor=beta0_factor, max_iter=20)
    exp.methods = [
        'ibcc'
    ]
    exp.run_methods(new_data=regen_data)

    # -------------------------------------------------------------------------
    beta0_factor = 0.1
    alpha0_diags = 0.1
    alpha0_factor = 0.1
    output_dir = os.path.join(evaluation.experiment.output_root_dir, 'arg_LMU_%f_%f_%f' % (beta0_factor, alpha0_diags, alpha0_factor))
    exp = Experiment(output_dir, nclasses, annos, gt, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev,
                     beta0_factor=beta0_factor, max_iter=20)
    exp.methods =  [
        'HMM_crowd',
    ]
    exp.run_methods(new_data=regen_data)

    # -------------------------------------------------------------------------
    beta0_factor = 1
    alpha0_diags = 0.1
    alpha0_factor = 0.1
    output_dir = os.path.join(evaluation.experiment.output_root_dir, 'arg_LMU_%f_%f_%f' % (beta0_factor, alpha0_diags, alpha0_factor))
    exp = Experiment(output_dir, nclasses, annos, gt, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev,
                     beta0_factor=beta0_factor, max_iter=20)
    exp.methods =  [
        'bac_acc_integrateIF',
    ]
    exp.run_methods(new_data=regen_data)

    # -------------------------------------------------------------------------
    beta0_factor = 0.1
    alpha0_diags = 1
    alpha0_factor = 100
    output_dir = os.path.join(evaluation.experiment.output_root_dir, 'arg_LMU_%f_%f_%f' % (beta0_factor, alpha0_diags, alpha0_factor))
    exp = Experiment(output_dir, nclasses, annos, gt, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev,
                     beta0_factor=beta0_factor, max_iter=20)
    exp.methods = [
        'bac_mace_integrateIF',
    ]

    exp.run_methods(new_data=regen_data)

    # -------------------------------------------------------------------------
    beta0_factor = 0.1
    alpha0_diags = 10
    alpha0_factor = 1
    output_dir = os.path.join(evaluation.experiment.output_root_dir, 'arg_LMU_%f_%f_%f' % (beta0_factor, alpha0_diags, alpha0_factor))
    exp = Experiment(output_dir, nclasses, annos, gt, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev,
                     beta0_factor=beta0_factor, max_iter=20)
    exp.methods =  [
        'bac_vec_integrateIF',
    ]

    exp.run_methods(new_data=regen_data)

    # -------------------------------------------------------------------------
    beta0_factor = 0.1
    alpha0_diags = 0.1
    alpha0_factor = 10
    output_dir = os.path.join(evaluation.experiment.output_root_dir, 'arg_LMU_%f_%f_%f' % (beta0_factor, alpha0_diags, alpha0_factor))
    exp = Experiment(output_dir, nclasses, annos, gt, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev,
                     beta0_factor=beta0_factor, max_iter=20)
    exp.methods = [
        'bac_ibcc_integrateIF',
        'bac_ibcc',
        'bac_ibcc_integrateIF_noHMM',
    ]

    exp.run_methods(new_data=regen_data)

    # -------------------------------------------------------------------------
    beta0_factor = 0.1
    alpha0_diags = 10
    alpha0_factor = 1
    output_dir = os.path.join(evaluation.experiment.output_root_dir, 'arg_LMU_%f_%f_%f' % (beta0_factor, alpha0_diags, alpha0_factor))
    exp = Experiment(output_dir, nclasses, annos, gt, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev,
                     beta0_factor=beta0_factor, max_iter=20, begin_factor=5.0)
    exp.methods = [
        'bac_seq_integrateIF',
        # 'bac_seq',
        # 'bac_seq_integrateIF_noHMM',
    ]

    exp.run_methods(new_data=regen_data)
