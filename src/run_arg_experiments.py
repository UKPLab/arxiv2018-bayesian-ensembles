from baselines.majority_voting import MajorityVoting
from data import data_utils
from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np
import pandas as pd

output_dir = '../../data/bayesian_sequence_combination/output/arg_LMU/'

regen_data = True

def load_arg_sentences(debug_size=0):
    data_dir = '../../data/bayesian_sequence_combination/data/argmin_LMU/'

    expert_file = data_dir + 'expert.csv'

    gold_text = pd.read_csv(expert_file, sep=',', usecols=[5]).values
    gold_doc_start = pd.read_csv(expert_file, sep=',', usecols=[1]).values
    expert_labels = pd.read_csv(expert_file, sep=',', usecols=[2,3,4]).values

    mv = MajorityVoting(expert_labels, 5)
    gold, gold_probs = mv.run()
    gold = data_utils.postprocess(gold, gold_doc_start)

    crowd_on_gold_file = data_dir + 'crowd_on_expert_labelled_sentences.csv'
    crowd_on_gold = pd.read_csv(crowd_on_gold_file, sep=',', usecols=range(2,28)).values

    crowd_no_gold_file = data_dir + 'crowd_on_sentences_with_no_experts.csv'
    crowd_without_gold = pd.read_csv(crowd_no_gold_file, sep=',', usecols=range(2,28)).values
    nogold_doc_start = pd.read_csv(crowd_no_gold_file, sep=',', usecols=[1]).values
    nogold_text = pd.read_csv(crowd_no_gold_file, sep=',', usecols=[28]).values

    # some of these data points have no annotations at all
    valididxs = np.any(crowd_without_gold != -1, axis=1)
    crowd_without_gold = crowd_without_gold[valididxs, :]
    nogold_doc_start = nogold_doc_start[valididxs]
    nogold_text = nogold_text[valididxs]

    N_nogold = crowd_without_gold.shape[0]

    # we need to flip 3 and 4 to fit out scheme here
    ICon_idxs = gold == 4
    BCon_idxs = gold == 3
    gold[ICon_idxs] = 3
    gold[BCon_idxs] = 4

    gt = np.concatenate((gold, np.zeros(N_nogold) - 1 ))
    annos = np.concatenate((crowd_on_gold, crowd_without_gold))

    ICon_idxs = annos == 4
    BCon_idxs = annos == 3
    annos[ICon_idxs] = 3
    annos[BCon_idxs] = 4

    doc_start = np.concatenate((gold_doc_start, nogold_doc_start))
    text = np.concatenate((gold_text, nogold_text))

    if debug_size:
        gt = gt[:debug_size]
        annos = annos[:debug_size]
        doc_start = doc_start[:debug_size]
        text = text[:debug_size]

    return gt, annos, doc_start, text

gt, annos, doc_start, text = load_arg_sentences()

exp = Experiment(None, 5, annos.shape[1], None, max_iter=20)

exp.save_results = True
exp.opt_hyper = False #True

best_bac_wm = 'bac_seq' # choose model with best score for the different BAC worker models
best_nu0factor = 0.1
best_diags = 1
best_factor = 1

exp.nu0_factor = best_nu0factor
exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor

nu_factors = [0.1, 10, 100]
diags = [0.1, 1, 10, 100] #, 50, 100]#[1, 50, 100]#[1, 5, 10, 50]
factors = [0.1, 1, 10]

# # run all the methods that don't require tuning here
exp.methods =  [
                'majority',
                'mace',
                'ds',
                'ibcc',
                'best',
                'worst',
                'bac_seq_integrateIF',
                'HMM_crowd',
]

# this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
exp.run_methods(annos, gt, doc_start, output_dir, text, new_data=regen_data)

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