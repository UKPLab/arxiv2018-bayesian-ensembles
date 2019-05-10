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

output_dir = '../../data/bayesian_sequence_combination/output/arg_LMU_corrected_gold_2/'

# TODO try the simple BIO task as well as 5-class thing

def cap_number_of_workers(crowd, doc_start, max_workers_per_doc):
    print('Reducing number of workers per document to %i' % max_workers_per_doc)
    doc_start_idxs = np.where(doc_start)[0]
    for d, doc in enumerate(doc_start_idxs):
        valid_workers = crowd[doc] != -1
        worker_count = np.sum(valid_workers)
        if worker_count > max_workers_per_doc:
            valid_workers = np.argwhere(valid_workers).flatten()
            drop_workers = valid_workers[max_workers_per_doc:]
            # print('dropping workers %s' % str(drop_workers))
            if d+1 < len(doc_start_idxs):
                next_doc = doc_start_idxs[d + 1]
                crowd[doc:next_doc, drop_workers] = -1
            else:
                crowd[doc:, drop_workers] = -1

    used_workers = np.any(crowd != -1, 0)
    crowd = crowd[:, used_workers]

    return crowd

def split_dev_set(gt, crowd, text, doc_start):
    all_doc_starts = np.where(doc_start)[0]
    doc_starts = np.where(doc_start & (gt != -1))[0]
    dev_starts = doc_starts[100:]

    crowd_dev = []
    gt_dev = []
    text_dev = []
    doc_start_dev = []
    for dev_start in dev_starts:
        next_doc_start = np.where(all_doc_starts == dev_start)[0][0] + 1
        if next_doc_start < all_doc_starts.shape[0]:
            doc_end = all_doc_starts[next_doc_start]
        else:
            doc_end = all_doc_starts.shape[0]

        crowd_dev.append(crowd[dev_start:doc_end])

        gt_dev.append(np.copy(gt[dev_start:doc_end]))
        gt[dev_start:doc_end] = -1

        text_dev.append(text[dev_start:doc_end])
        doc_start_dev.append(doc_start[dev_start:doc_end])

    crowd_dev = np.concatenate(crowd_dev, axis=0)
    gt_dev = np.concatenate(gt_dev, axis=0)
    text_dev = np.concatenate(text_dev, axis=0)
    doc_start_dev = np.concatenate(doc_start_dev, axis=0)

    return gt, crowd_dev, gt_dev, doc_start_dev, text_dev

def load_arg_sentences(debug_size=0, regen_data=False, second_batch_workers_only=False, gold_labelled_only=False,
                       max_workers_per_doc=5):
    data_dir = '../../data/bayesian_sequence_combination/data/argmin_LMU/'

    if not regen_data and os.path.exists(data_dir + 'evaluation_gold.csv'):
        #reload the data for the experiments from cache files
        gt = pd.read_csv(data_dir + 'evaluation_gold.csv', usecols=[1]).values.astype(int)
        crowd = pd.read_csv(data_dir + 'evaluation_crowd.csv').values[:, 1:].astype(int)
        doc_start = pd.read_csv(data_dir + 'evaluation_doc_start.csv', usecols=[1]).values.astype(int)
        text = pd.read_csv(data_dir + 'evaluation_text.csv', usecols=[1]).values

        if second_batch_workers_only:
            crowd = crowd[:, 26:]

        if gold_labelled_only:
            idxs = gt.flatten() != -1
            gt = gt[idxs]
            crowd = crowd[idxs, :]
            doc_start = doc_start[idxs]
            text = text[idxs]

        if max_workers_per_doc > 0:
            crowd = cap_number_of_workers(crowd, doc_start, max_workers_per_doc)

        # split dev set
        gt, crowd_dev, gt_dev, doc_start_dev, text_dev = split_dev_set(gt, crowd, text, doc_start)

        return gt, crowd, doc_start, text, crowd_dev, gt_dev, doc_start_dev, text_dev

    expert_file = data_dir + 'expert_corrected_disagreements.csv'

    gold_text = pd.read_csv(expert_file, sep=',', usecols=[6]).values
    gold_doc_start = pd.read_csv(expert_file, sep=',', usecols=[1]).values
    gold = pd.read_csv(expert_file, sep=',', usecols=[7]).values.astype(int).flatten()

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

    if gold_labelled_only:
        idxs = gt.flatten() != -1
        gt = gt[idxs]
        crowd = crowd[idxs, :]
        doc_start = doc_start[idxs]
        text = text[idxs]

    gt = gt.astype(int)

    if max_workers_per_doc > 0:
        crowd = cap_number_of_workers(crowd, doc_start, max_workers_per_doc)

    gt, crowd_dev, gt_dev, doc_start_dev, text_dev = split_dev_set(gt, crowd, text, doc_start)

    return gt, crowd, doc_start, text, crowd_dev, gt_dev, doc_start_dev, text_dev

if __name__ == '__main__':

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

    N = 0 #4521 # set to 0 to use all
    gt, annos, doc_start, text, annos_dev, gt_dev, doc_start_dev, text_dev = load_arg_sentences(
        N, regen_data, second_batch_workers_only, gold_labelled_only)
    N = float(len(gt))

    valid_workers = np.any(annos != -1, axis=0)
    print('Valid workers for this subset are %s' % str(np.argwhere(valid_workers).flatten()))

    nclasses = 5

    # ------------------------------------------------------------------------------------------------------------------

    nu_factors = [0.1, 1, 10]
    diags = [0.1, 1, 10, 100]
    factors = [0.1, 1, 10, 100]

    #  ------------------------------------------------------------------------------------------------

    exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=20)

    methods_to_tune = [
                    # 'ibcc',
                    # 'bac_seq_integrateIF',
                    # 'bac_vec_integrateIF',
                    # 'HMM_crowd',
                    'bac_ibcc_integrateIF',
                    'bac_acc_integrateIF',
                    'bac_mace_integrateIF',
                    ]

    for m, method in enumerate(methods_to_tune):
        print('TUNING %s' % method)

        best_scores = exp.tune_alpha0(diags, factors, nu_factors, method, annos_dev, gt_dev, doc_start_dev,
                                      output_dir, text_dev, metric_idx_to_optimise=11, new_data=regen_data)
        best_idxs = best_scores[1:].astype(int)
        exp.nu0_factor = nu_factors[best_idxs[0]]
        exp.alpha0_diags = diags[best_idxs[1]]
        exp.alpha0_factor = factors[best_idxs[2]]

        print('Best values for %s: %f, %f, %f' % (method, exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))

        # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
        exp.methods = [method]
        exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, return_model=True,
                    ground_truth_val=gt_dev, doc_start_val=doc_start_dev, text_val=text_dev,
                    new_data=regen_data
        )

        if m == 0:
            bsc_ibcc_0 = best_scores[1]
            bsc_ibcc_1 = best_scores[2]
            bsc_ibcc_2 = best_scores[3]


    # ------------------------------------------------------------------------------------------------------------------

    exp = Experiment(None, nclasses, annos.shape[1], None, max_iter=50)

    exp.save_results = True
    exp.opt_hyper = False #True

    # values obtained from tuning on dev:
    best_nu0factor = 10#1
    best_diags = 100
    best_factor = 10#100

    exp.nu0_factor = best_nu0factor
    exp.alpha0_diags = best_diags
    exp.alpha0_factor = best_factor

    exp.methods =  [
        # 'bac_seq_integrateIF',
        'bac_seq',
        'bac_seq_integrateIF_noHMM',
    ]

    # exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, test_no_crowd=False)

    # values obtained from tuning on dev:
    best_nu0factor = 0.1
    best_diags = 10
    best_factor = 0.1

    exp.nu0_factor = best_nu0factor
    exp.alpha0_diags = best_diags
    exp.alpha0_factor = best_factor

    exp.methods =  [
        'bac_vec_integrateIF',
    ]

    exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, test_no_crowd=False)

    best_nu0factor = 1
    best_diags = 0.1
    best_factor = 0.1

    exp.nu0_factor = best_nu0factor
    exp.alpha0_diags = best_diags
    exp.alpha0_factor = best_factor

    exp.methods =  [
        'bac_mace_integrateIF',
        'bac_acc_integrateIF',
    ]

    # exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, test_no_crowd=False)

    # settings obtained by tuning on dev:
    best_nu0factor = 0.1 # 1
    best_diags = 100 # 0.1
    best_factor = 0.1 # 0.1

    exp.nu0_factor = best_nu0factor
    exp.alpha0_diags = best_diags
    exp.alpha0_factor = best_factor

    exp.methods =  [
        # 'bac_ibcc_integrateIF',
        'bac_ibcc',
        'bac_ibcc_integrateIF_noHMM',
    ]

    exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, test_no_crowd=False)

    # settings obtained from tuning on dev:
    best_nu0factor = 0.1
    best_diags = 0.1
    best_factor = 1.0

    exp.nu0_factor = best_nu0factor
    exp.alpha0_diags = best_diags
    exp.alpha0_factor = best_factor

    exp.methods =  [
                    'majority',
                    'mace',
                    'ds',
                    'ibcc',
                    'best',
                    'worst',
                    # 'bac_seq_integrateIF_weakprior',
                    # 'bac_ibcc_integrateIF_weakprior',
                    # 'bac_vec_integrateIF_weakprior',
    ]

    exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, new_data=True)

    # settings obtained from tuning on dev:
    best_nu0factor = 0.1
    best_diags = 0.1
    best_factor = 0.1

    exp.nu0_factor = best_nu0factor
    exp.alpha0_diags = best_diags
    exp.alpha0_factor = best_factor

    exp.methods =  [
        # 'bac_vec_integrateIF',
        'HMM_crowd',
    ]

    # exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, new_data=True)