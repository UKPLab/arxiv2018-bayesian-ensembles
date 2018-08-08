'''
Created on April 27, 2018

determine whether the confusion matrix for the LSTM is reasonable. -- currently running with atEnd computation,
same priors as workers, to save conf mats as npy file -- now looks okay but results are still same.

The performance drop when integrating the LSTM should not occur if the model is set up correctly,
since the LSTM should reinforce the existing predictions in most cases. It may cause a problem if
its labels push the E_t values towards one for data points that were
previously uncertain. This could, in turn, change the sequence.
But it's not clear why that wouldn't simply tend toward the LSTM's predictions instead -- perhaps
in cases where there are multiple possible sequences, we get a mishmash of both?
TODO: error analysis to see what the differences between the integratedLSTM, thenLSTM, and BSC predictions are.
A solution would be to soften all confusion matrices
by increasing the alpha0 factor. This should be tuned on the validation set.

Overfitting the LSTM should reinforce the existing predictions as mentioned above.
However, underfitting should not be an issue as it was not an issue for thenLSTM.

Another possible problem may be the confusion matrices -- is there something strange in there for some rare label
transitions? Strong priors should help here -- use validation set to tune.

Could also try the LSTM with totally flat prior (no disallowed counts etc).

Using a flat prior with very high counts does stop the LSTM from affecting results much.
Results with very informative, reliable prior should be like thenLSTM, but are still extremely similar
to the results with soft priors on the LSTM. Perhaps over-confidence in the label distributions is causing
some of the sequence labels to change. E.g. O->I

Check results with moderate counts for a flat prior - tune on validation -- validation didn't work well with the small training set.
Exclude the BOF part when using +LSTM because the text features shouldn't get used twice? -- this does seem to help.
Tried: try more iterations to allow proper convergence? -- does not help.

Fixed: comparison between results on air and krusty -- still same? Numerical differences? Did our latest prior changes
reduce performance from the values in the table? After this is dones, go through the ideas below.
- Current priors have reduced f1 score by 1% over the values in the table. Find the date of these results and find out
how priors were previously set.
- In current setup, f1scores are only slightly lower than those without the LSTM
- However, on the test data, the performance has decreased further by just over one percent (may be due to priors again)
- We aim to get better performance on the test data than with HMMCrowd or non-integrated method, even if no improvement on training data

run with prior biasing away from outside labels. -- ran, it caused big problems, except with bias set to one did little damage.

Run with one iteration per VB iteration -- did very poorly
Can we get LSTM to give us probabilities when integrated? -- tried, didn't make much difference, maybe worse when running at end?
MaybeTODO: Can we train LSTM on probabilities?

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

output_dir = '../../data/bayesian_annotator_combination/output/ner-by-sentence/'

regen_data = False
gt, annos, doc_start, text, gt_nocrowd, doc_start_nocrowd, text_nocrowd, gt_task1_val, gt_val, doc_start_val, text_val, _ = \
    load_data.load_ner_data(regen_data)

exp = Experiment(None, 9, annos.shape[1], None, max_iter=20)
exp.save_results = True
exp.opt_hyper = False#True

best_bac_wm = 'bac_seq' #'unknown' # choose model with best score for the different BAC worker models
best_bac_wm_score = -np.inf

best_nu0factor = 0.1
best_diags = 1
best_factor = 1
best_acc_bias = 0

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor
exp.alpha0_acc_bias = best_acc_bias

exp.alpha0_diags_lstm = 0.1
exp.alpha0_factor_lstm = 0.1

nu_factors = [0.1, 10, 100]
diags = [0.1, 1, 10, 100]
factors = [0.1, 1, 10]
# acc_biases = [1, 10, 100]

methods_to_tune = [
                   #'ibcc',
                   'bac_vec_integrateBOF',
                   'bac_ibcc_integrateBOF',
                   'bac_seq_integrateBOF',
                   'bac_acc_integrateBOF',
                   'bac_mace_integrateBOF'
                   ]

# tune with small dataset to save time
s = 250
idxs = np.argwhere(gt_task1_val != -1)[:, 0]
ndocs = np.sum(doc_start[idxs])

if ndocs > s:
    idxs = idxs[:np.argwhere(np.cumsum(doc_start[idxs])==s)[0][0]]
elif ndocs < s:  # not enough validation data
    moreidxs = np.argwhere(gt != -1)[:, 0]
    deficit = s - ndocs
    ndocs = np.sum(doc_start[moreidxs])
    if ndocs > deficit:
        moreidxs = moreidxs[:np.argwhere(np.cumsum(doc_start[moreidxs])==deficit)[0][0]]
    idxs = np.concatenate((idxs, moreidxs))

tune_gt = gt[idxs]
tune_annos = annos[idxs]
tune_doc_start = doc_start[idxs]
tune_text = text[idxs]
tune_gt_task1_val = gt_task1_val[idxs]

for m, method in enumerate(methods_to_tune):
    print('TUNING %s' % method)

    best_scores = exp.tune_alpha0(diags, factors, nu_factors, method, tune_annos, tune_gt_task1_val, tune_doc_start,
                                  output_dir, tune_text)

    # best_scores = exp.tune_alpha0(lstm_diags, lstm_factors, nu_factors, method, tune_annos, tune_gt_task1_val, tune_doc_start,
    #                               output_dir, tune_text, tune_lstm=True,
    #                               ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val)

    best_idxs = best_scores[1:].astype(int)
    exp.nu0_factor = nu_factors[best_idxs[0]]
    exp.alpha0_diags = diags[best_idxs[1]]
    exp.alpha0_factor = factors[best_idxs[2]]

    # exp.alpha0_diags_lstm = diags[best_idxs[1]]
    # exp.alpha0_factor_lstm = factors[best_idxs[2]]

    print('Best values: %f, %f, %f' % (exp.nu0_factor, exp.alpha0_diags, exp.alpha0_factor))

    # best_score, best_idx = exp.tune_acc_bias(acc_biases, method, tune_annos, tune_gt_task1_val, tune_doc_start,
    #                               output_dir, tune_text)

    # this will run task 1 -- train on all crowdsourced data, test on the labelled portion thereof
    exp.methods = [method]
    exp.run_methods(annos, gt, doc_start, output_dir, text, rerun_all=True, return_model=False,
                ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
                ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd,
                text_nocrowd=text_nocrowd,
                new_data=regen_data
                )

    best_score = best_scores[0]
    if 'bac_seq' in method and best_score > best_bac_wm_score:
        best_bac_wm = 'bac_' + method.split('_')[1]
        best_bac_wm_score = best_score
        best_diags = exp.alpha0_diags
        best_factor = exp.alpha0_factor
        best_nu0factor = exp.nu0_factor
    #
    # if 'bac_seq' in method and best_score > best_bac_wm_score:
    #     best_bac_wm_score = best_score
    #     best_acc_bias = exp.alpha0_acc_bias
    #     best_bac_wm = 'bac_' + method.split('_')[1]

print('best BAC method tested here = %s' % best_bac_wm)

exp.alpha0_diags = best_diags
exp.alpha0_factor = best_factor
exp.nu0_factor = best_nu0factor
exp.alpha0_acc_bias = best_acc_bias

# run all the methods that don't require tuning here
exp.methods =  [
                # 'majority',
                # 'mace',
                # 'ds',
                # 'best',
                # 'worst',
                best_bac_wm,
]

# should run both task 1 and 2.
exp.run_methods(
    annos, gt, doc_start, output_dir, text,
    ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
    ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
    new_data=regen_data
)

# # reset to free memory? ------------------------------------------------------------------------------------------------
# exp = Experiment(None, 9, annos.shape[1], None, alpha0_factor=16, alpha0_diags=1, max_iter=20)
# exp.save_results = True
# exp.opt_hyper = False#True
#
# exp.alpha0_diags = best_diags
# exp.alpha0_factor = best_factor
# exp.nu0_factor = best_nu0factor
# exp.alpha0_acc_bias = best_acc_bias
#
# # run all the methods that don't require tuning here
# exp.methods =  [
#                 'HMM_crowd',
# ]
#
# # should run both task 1 and 2.
# exp.run_methods(
#     annos, gt, doc_start, output_dir, text,
#     ground_truth_val=gt_val, doc_start_val=doc_start_val, text_val=text_val,
#     ground_truth_nocrowd=gt_nocrowd, doc_start_nocrowd=doc_start_nocrowd, text_nocrowd=text_nocrowd,
#     new_data=regen_data
# )