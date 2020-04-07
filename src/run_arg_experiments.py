import os
import sys
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

    N = 0 #4521 # set to 0 to use all, any other number will debug on a subset
    gt, annos, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev = load_data.load_arg_sentences(
        N, regen_data, second_batch_workers_only, gold_labelled_only)
    N = float(len(gt))

    valid_workers = np.any(annos != -1, axis=0)
    print('Valid workers for this subset are %s' % str(np.argwhere(valid_workers).flatten()))

    nclasses = 5

    # Run with hyperparameter tuning -------------------------------------------------------

    # TUNE on F1 score.
    # If multiple hyperparameters produce same F1, we could use CEE as a tie-breaker?

    # nu_factors = [0.1, 1, 10]
    # diags = [0.1, 1, 10]
    # factors = [0.1, 1, 10, 100]
    #
    # output_dir = os.path.join(load_data.output_root_dir, 'arg_LMU_strictF1')
    # exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=20, begin_factor=5.0)
    #
    # methods_to_tune = [
    #     'ibcc',
    #     'bac_ibcc_integrateIF',
    #     'bac_seq_integrateIF',
    #     'bac_mace_integrateIF',
    #     'ds',
    #     'bac_vec_integrateIF',
    # ]
    #
    # for m, method in enumerate(methods_to_tune):
    #     print('TUNING %s' % method)
    #
    #     best_scores = exp.tune_alpha0(diags, factors, nu_factors, method, annos_dev, gt_dev, doc_start_dev,
    #                                   output_dir, text_dev, metric_idx_to_optimise=8, new_data=regen_data)
    #     best_idxs = best_scores[1:].astype(int)
    #     exp.nu0_factor = nu_factors[best_idxs[0]]
    #     exp.alpha0_diags = diags[best_idxs[1]]
    #     exp.alpha0_factor = factors[best_idxs[2]]
    #
    #     print('Best values for %s: %f, %f, %f' % (method, exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
    #
    #     # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
    #     exp.methods = [method]
    #     exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, return_model=True,
    #                     ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
    #                     new_data=regen_data
    #                     )

    # Run with optimal values found during our experiments by tuning on dev set ---------------------------------


    # exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=20)
    # exp.nu0_factor = 0.1 # this is used by DS but not the other methods here
    # output_dir = os.path.join(load_data.output_root_dir, 'arg_LMU_%f' % exp.nu0_factor)
    #
    # exp.methods =  [
    #     # 'majority',
    #     # 'best',
    #     # 'worst'
    #     'ds'
    # ]
    # exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, test_no_crowd=False)


    # exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=20)
    # exp.nu0_factor = 10
    # exp.alpha0_diags = 0.1
    # exp.alpha0_factor = 10
    # # 1, 1, 10 gives similar results -- not optimal on dev, but was tried as default values before tuning...
    # output_dir = os.path.join(load_data.output_root_dir, 'arg_LMU_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
    #
    # exp.methods = [
    #     'ibcc'
    # ]
    # exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, test_no_crowd=False)


    # exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=20)
    # exp.nu0_factor = 0.1
    # exp.alpha0_diags = 0.1
    # exp.alpha0_factor = 0.1
    # output_dir = os.path.join(load_data.output_root_dir, 'arg_LMU_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
    #
    # exp.methods =  [
    #     'HMM_crowd',
    # ]
    #
    # exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, new_data=True)


    # exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=20)
    # exp.nu0_factor = 1
    # exp.alpha0_diags = 0.1
    # exp.alpha0_factor = 0.1
    # output_dir = os.path.join(load_data.output_root_dir, 'arg_LMU_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
    #
    # exp.methods =  [
    #     'bac_acc_integrateIF',
    # ]
    #
    # exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, new_data=True)


    # exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=20)
    # exp.nu0_factor = 0.1
    # exp.alpha0_diags = 1
    # exp.alpha0_factor = 100
    # output_dir = os.path.join(load_data.output_root_dir, 'arg_LMU_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
    #
    # exp.methods =  [
    #     'bac_mace_integrateIF',
    # ]
    #
    # exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, new_data=True)


    # exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=20)
    # exp.nu0_factor = 0.1
    # exp.alpha0_diags = 10
    # exp.alpha0_factor = 1
    # output_dir = os.path.join(load_data.output_root_dir, 'arg_LMU_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
    #
    # exp.methods =  [
    #     'bac_vec_integrateIF',
    # ]
    #
    # exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, new_data=True)


    # exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=20)
    # exp.nu0_factor = 0.1
    # exp.alpha0_diags = 0.1
    # exp.alpha0_factor = 10
    # output_dir = os.path.join(load_data.output_root_dir, 'arg_LMU_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))
    #
    # exp.methods =  [
    #     'bac_ibcc_integrateIF',
    #     'bac_ibcc',
    #     'bac_ibcc_integrateIF_noHMM',
    # ]
    #
    # exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, new_data=True)


    exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=20, begin_factor=5.0)
    exp.nu0_factor = 0.1
    exp.alpha0_diags = 10
    exp.alpha0_factor = 1
    output_dir = os.path.join(load_data.output_root_dir, 'arg_LMU_%f_%f_%f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))

    exp.methods =  [
        'bac_seq_integrateIF',
        # 'bac_seq',
        # 'bac_seq_integrateIF_noHMM',
    ]

    exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, new_data=True)
