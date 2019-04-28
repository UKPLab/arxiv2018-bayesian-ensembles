import os
import sys

from baselines.ibcc import IBCC
from baselines.majority_voting import MajorityVoting
from bsc.bsc import BSC
from data import data_utils
from evaluation.experiment import Experiment, calculate_scores
import data.load_data as load_data
import numpy as np
import pandas as pd

output_dir = '../../data/bayesian_sequence_combination/output/arg_LMU_corrected_gold/'

rerun_all = True

# TODO try the simple BIO task as well as 5-class thing

def load_arg_sentences(debug_size=0, regen_data=False, second_batch_workers_only=False):
    data_dir = '../../data/bayesian_sequence_combination/data/argmin_LMU/'

    if not regen_data and os.path.exists(data_dir + 'evaluation_gold.csv'):
        #reload the data for the experiments from cache files
        gt = pd.read_csv(data_dir + 'evaluation_gold.csv', usecols=[1]).values.astype(int)
        crowd = pd.read_csv(data_dir + 'evaluation_crowd.csv').values[:, 1:].astype(int)
        doc_start = pd.read_csv(data_dir + 'evaluation_doc_start.csv', usecols=[1]).values.astype(int)
        text = pd.read_csv(data_dir + 'evaluation_text.csv', usecols=[1]).values

        if second_batch_workers_only:
            crowd = crowd[:, 26:]

        # idxs = gt.flatten() != -1
        # gt = gt[idxs]
        # crowd = crowd[idxs, :]
        # doc_start = doc_start[idxs]
        # text = text[idxs]

        return gt, crowd, doc_start, text

    expert_file = data_dir + 'expert_corrected_disagreements.csv'

    gold_text = pd.read_csv(expert_file, sep=',', usecols=[6]).values
    gold_doc_start = pd.read_csv(expert_file, sep=',', usecols=[1]).values
    gold = pd.read_csv(expert_file, sep=',', usecols=[7]).values.flatten()

    crowd_on_gold_file = data_dir + 'crowd_on_expert_labelled_sentences.csv'
    crowd_on_gold = pd.read_csv(crowd_on_gold_file, sep=',', usecols=range(2,28)).values

    crowd_no_gold_file = data_dir + 'crowd_on_sentences_with_no_experts.csv'
    crowd_without_gold = pd.read_csv(crowd_no_gold_file, sep=',', usecols=range(2,100)).values
    nogold_doc_start = pd.read_csv(crowd_no_gold_file, sep=',', usecols=[1]).values
    nogold_text = pd.read_csv(crowd_no_gold_file, sep=',', usecols=[100]).values

    print('Number of tokens = %i' % crowd_without_gold.shape[0])

    # some of these data points may have no annotations
    valididxs = np.any(crowd_without_gold != -1, axis=1)
    crowd_without_gold = crowd_without_gold[valididxs, :]
    doc_start = nogold_doc_start[valididxs]
    text = nogold_text[valididxs]

    print('Number of crowd-labelled tokens = %i' % crowd_without_gold.shape[0])

    N = crowd_without_gold.shape[0]

    # now line up the gold sentences with the complete set of crowd data
    crowd = np.zeros((N, crowd_on_gold.shape[1] + crowd_without_gold.shape[1]), dtype=int) - 1
    crowd[:, crowd_on_gold.shape[1]:] = crowd_without_gold

    # crowd_labels_present = np.any(crowd != -1, axis=1)
    # N_withcrowd = np.sum(crowd_labels_present)

    gt = np.zeros(N) - 1

    gold_docs = np.split(gold_text, np.where(gold_doc_start == 1)[0][1:], axis=0)
    gold_gold = np.split(gold, np.where(gold_doc_start == 1)[0][1:], axis=0)
    gold_crowd = np.split(crowd_on_gold, np.where(gold_doc_start == 1)[0][1:], axis=0)

    nogold_docs = np.split(text, np.where(nogold_doc_start == 1)[0][1:], axis=0)
    for d, doc in enumerate(gold_docs):

        print('matching gold doc %i of %i' % (d, len(gold_docs)))

        loc_in_nogold = 0

        for doc_nogold in nogold_docs:
            if np.all(doc == doc_nogold):
                len_doc_nogold = len(doc_nogold)
                break
            else:
                loc_in_nogold += len(doc_nogold)

        locs_in_nogold = np.arange(loc_in_nogold, len_doc_nogold+loc_in_nogold)
        gt[locs_in_nogold] = gold_gold[d]
        crowd[locs_in_nogold, :crowd_on_gold.shape[1]] = gold_crowd[d]

    # we need to flip 3 and 4 to fit our scheme here
    ICon_idxs = gt == 4
    BCon_idxs = gt == 3
    gt[ICon_idxs] = 3
    gt[BCon_idxs] = 4

    ICon_idxs = crowd == 4
    BCon_idxs = crowd == 3
    crowd[ICon_idxs] = 3
    crowd[BCon_idxs] = 4

    if debug_size:
        gt = gt[:debug_size]
        crowd = crowd[:debug_size]
        doc_start = doc_start[:debug_size]
        text = text[:debug_size]

    # save files for our experiments with the tag 'evaluation_'
    pd.DataFrame(gt).to_csv(data_dir + 'evaluation_gold.csv')
    pd.DataFrame(crowd).to_csv(data_dir + 'evaluation_crowd.csv')
    pd.DataFrame(doc_start).to_csv(data_dir + 'evaluation_doc_start.csv')
    pd.DataFrame(text).to_csv(data_dir + 'evaluation_text.csv')

    if second_batch_workers_only:
        crowd = crowd[:, 26:]

    return gt, crowd, doc_start, text

if __name__ == '__main__':

    if len(sys.argv) > 1:
        second_batch_workers_only = bool(sys.argv[1])
    else:
        second_batch_workers_only = False

    if len(sys.argv) > 2:
        run_on_all = bool(sys.argv[2])
    else:
        run_on_all = False

    print('Running ' + ('with' if second_batch_workers_only else 'without') + ' second-batch workers only.')

    N = 0 #4521 # set to 0 to use all
    gt, annos, doc_start, text = load_arg_sentences(N, False, second_batch_workers_only) #4521 will train only on the gold-labelled stuff
    N = float(len(gt))

    valid_workers = np.any(annos != -1, axis=0)
    print('Valid workers for this subset are %s' % str(np.argwhere(valid_workers).flatten()))

    nclasses = 5

    if run_on_all:
        # produce a gold standard using IBCC for comparison ----------------------------------------------------------------
        best_nu0factor = 1.0
        best_diags = 1.0
        best_factor = 1.0

        ibcc_alpha0 = best_factor * np.ones((nclasses, nclasses)) + best_diags * np.eye(nclasses)
        ibc = IBCC(nclasses=nclasses, nscores=nclasses, nu0=np.ones(nclasses) * best_nu0factor, alpha0=ibcc_alpha0,
                   uselowerbound=False)
        ibc.verbose = True
        ibc.max_iterations = 20
        probs = ibc.combine_classifications(annos, table_format=True, goldlabels=gt.flatten())  # posterior class probabilities
        agg = probs.argmax(axis=1)  # aggregated class labels

        # test the performance of the predictions -- this means evaluating on the training set as a sanity check
        result, _ = calculate_scores(nclasses, False, agg, gt.flatten(), probs, doc_start,
                                     bootstrapping=True, print_per_class_results=True)

        pd.DataFrame(result, columns=['IBCC_GOLD']).to_csv(output_dir + '/gold_ibcc.csv')

        # produce a gold standard using all available data -----------------------------------------------------------------
        best_nu0factor = N / nclasses * 0.1
        best_diags = N / nclasses * 0.1
        best_factor = N / nclasses * 0.1

        bsc_model = BSC(L=nclasses, K=annos.shape[1], max_iter=20, inside_labels=[0,3], outside_labels=[1],
                        beginning_labels=[2,4], alpha0_diags=best_diags, alpha0_factor=best_factor,
                        beta0_factor=best_nu0factor, exclusions=None, before_doc_idx=-1, worker_model='seq',
                        tagging_scheme='IOB2', data_model=[], transition_model='HMM', no_words=False)
        bsc_model.verbose = True
        bsc_model.max_iter = 20

        probs, agg = bsc_model.run(annos, doc_start, text, gold_labels=gt)

        # test the performance of the predictions -- this means evaluating on the training set as a sanity check
        result, _ = calculate_scores(nclasses, False, agg, gt.flatten(), probs, doc_start,
                                     bootstrapping=True, print_per_class_results=True)

        pd.DataFrame(result, columns=['BSC_SEQ_GOLD']).to_csv(output_dir + '/gold_bsc_seq.csv')

    # ------------------------------------------------------------------------------------------------------------------

    exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=20)

    exp.save_results = True
    exp.opt_hyper = False #True

    best_bac_wm = 'bac_seq' # choose model with best score for the different BAC worker models
    best_nu0factor = N / nclasses * 0.1
    best_diags = N / nclasses * 0.1
    best_factor = N / nclasses * 0.1

    exp.nu0_factor = best_nu0factor
    exp.alpha0_diags = best_diags
    exp.alpha0_factor = best_factor

    exp.methods =  [
                    'ibcc',
                    'bac_seq_integrateIF',
                    'bac_ibcc_integrateIF',
                    'bac_vec_integrateIF',
                    'bac_seq',
                    'bac_seq_integrateIF_noHMM',
    ]

    # exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=rerun_all)

    best_nu0factor = 1
    best_diags = 1
    best_factor = 1

    exp.nu0_factor = best_nu0factor
    exp.alpha0_diags = best_diags
    exp.alpha0_factor = best_factor

    exp.methods =  [
                    # 'majority',
                    # 'mace',
                    # 'ds',
                    # 'ibcc',
                    # 'best',
                    # 'worst',
                    # 'HMM_crowd',
                    'bac_seq_integrateIF_weakprior',
                    'bac_ibcc_integrateIF_weakprior',
                    'bac_vec_integrateIF_weakprior',
    ]

    exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=rerun_all)

    #
    # nu_factors = [0.1, 10, 100]
    # diags = [0.1, 1, 10, 100] #, 50, 100]#[1, 50, 100]#[1, 5, 10, 50]
    # factors = [0.1, 1, 10]
    #
    # #  ------------------------------------------------------------------------------------------------
    #
    # methods_to_tune = [
    #                 # 'ibcc',
    #                 'bac_seq_integrateIF',
    #                 # 'bac_vec_integrateIF',
    #                 # 'bac_ibcc_integrateIF',
    #                 # 'bac_acc_integrateIF',
    #                 # 'bac_mace_integrateIF',
    #                 # 'bac_ibcc',
    #                 # 'bac_ibcc_integrateIF_noHMM',
    #                 # 'bac_seq',
    #                 # 'bac_seq_integrateIF_noHMM',
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
    #
    # for m, method in enumerate(methods_to_tune):
    #     print('TUNING %s' % method)
    #
    #     best_scores = exp.tune_alpha0(diags, factors, nu_factors, method, tune_annos, tune_gt_dev, tune_doc_start,
    #                                   output_dir, tune_text, metric_idx_to_optimise=11)
    #     best_idxs = best_scores[1:].astype(int)
    #     exp.nu0_factor = nu_factors[best_idxs[0]]
    #     exp.alpha0_diags = diags[best_idxs[1]]
    #     exp.alpha0_factor = factors[best_idxs[2]]
    #
    #     print('Best values: %f, %f, %f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))

    #     # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
    #     exp.methods = [method]
    #     exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, return_model=True,
    #                 ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
    #                 new_data=regen_data
    #     )
    #
    # #     if 'bac_seq' in method and best_score > best_bac_wm_score:
    # #         best_bac_wm = method
    # #         best_bac_wm_score = best_score
    # #         best_diags = exp.alpha0_diags
    # #         best_factor = exp.alpha0_factor
    # #         best_nu0factor = exp.nu0_factor
    #