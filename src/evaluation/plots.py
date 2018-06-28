'''
Created on Nov 1, 2016

@author: Melvin Laux
'''

#import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)

import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

PARAM_NAMES = ['acc_bias',
               'miss_bias',
               'short_bias',
               'num_docs',
               'doc_length',
               'group_sizes',
               ]

SCORE_NAMES = ['accuracy',
               'precision-tokens',
               'recall-tokens',
               'f1-score-tokens',
               'auc-score',
               'cross-entropy-error',
               'precision-spans-strict',
               'recall-spans-strict',
               'f1-score-spans-strict',
               'precision-spans-relaxed',
               'recall-spans-relaxed',
               'f1-score-spans-relaxed',
               'count error',
               'number of invalid labels',
               'mean length error'
               ]

def make_plot(methods, param_idx, x_vals, y_vals, x_ticks_labels, ylabel, title='parameter influence'):

    styles = ['-', '--', '-.', ':']
    markers = ['o', 'v', 's', 'p', '*']

    for j in range(len(methods)):
        plt.plot(x_vals, np.mean(y_vals[:, j, :], 1), label=methods[j], ls=styles[j%4], marker=markers[j%5])

    plt.legend(loc='best')

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(PARAM_NAMES[param_idx])
    plt.xticks(x_vals, x_ticks_labels)
    plt.grid(True)
    plt.grid(True, which='Minor', axis='y')

    if np.min(y_vals) < 0:
        plt.ylim([np.min(y_vals), np.max([1, np.max(y_vals)])])
    else:
        plt.ylim([0, np.max([1, np.max(y_vals)])])


def plot_results(param_values, methods, param_idx, results, show_plot=False, save_plot=False, output_dir='/output/',
                 score_names=None, title='parameter influence'):
    # Plots the results for varying the parameter specified by param_idx

    param_values = np.array(param_values)

    if score_names is None:
        score_names = SCORE_NAMES

    # create output directory if necessary
    if save_plot and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initialise values for x axis
    if (param_values.ndim > 1):
        x_vals = param_values[:, 0]
    else:
        x_vals = param_values

    # initialise x-tick labels
    x_ticks_labels = list(map(str, param_values))

    for i in range(len(score_names)):
        make_plot(methods, param_idx, x_vals, results[:,i,:,:], x_ticks_labels, score_names[i], title)

        if save_plot:
            print('Saving plot...')
            plt.savefig(output_dir + 'plot_' + score_names[i] + '.png')
            plt.clf()

        if show_plot:
            plt.show()

        if i == 5:
            make_plot(methods, param_idx, x_vals, results[:,i,:,:], x_ticks_labels, score_names[i], title)
            plt.ylim([0,1])

            if save_plot:
                print('Saving plot...')
                plt.savefig(output_dir + 'plot_' + score_names[i] + '_zoomed' + '.png')
                plt.clf()

            if show_plot:
                plt.show()

def plot_active_learning_results(results_dir, output_dir):

    ndocs = np.array([605, 1210, 1815, 2420, 3025, 3630, 4235, 4840, 5445, 6045]) # NER dataset
    methods = np.array([
        'HMM_crowd_then_LSTM',
        'bac_seq_integrateBOF_then_LSTM',
        'bac_seq_integrateBOF_integrateLSTM_atEnd',
    ])

    # results: y-value, metric, methods, runs
    results = np.zeros((len(ndocs), len(SCORE_NAMES), len(methods), 1))
    results_nocrowd = np.zeros((len(ndocs), len(SCORE_NAMES), len(methods), 1))

    resfiles = os.listdir(results_dir)

    for resfile in resfiles:

        if resfile[:7] != 'result_':
            continue

        if 'result_std_started' in resfile:
            continue
        elif 'result_std_nocrowd' in resfile:
            continue

        res = pd.read_csv(os.path.join(results_dir, resfile))

        for col in res.columns:
            methodidx = np.argwhere(methods == col.strip("\\# '"))[0][0]
            ndocsidx = int(resfile.split('.csv')[0].split('Nseen')[-1])
            ndocsidx = np.argwhere(ndocs == ndocsidx)[0][0]

        if 'result_started' in resfile:
            results[ndocsidx, :, methodidx, 0] = res[col]
        elif 'result_nocrowd_started' in resfile:
            results_nocrowd[ndocsidx, :, methodidx, 0] = res[col]

    output_pool_dir = os.path.join(output_dir, 'pool/')
    output_test_dir = os.path.join(output_dir, 'test/')

    plot_results(ndocs, methods, 3, results, False, True, output_pool_dir, SCORE_NAMES,
                 title='Active Learning: Pool Data')
    plot_results(ndocs, methods, 3, results_nocrowd, False, True, output_test_dir, SCORE_NAMES,
                 title='Active Learning: Test Data')

if __name__ == '__main__':
    print('Plotting active learning results...')

    results_dir = '../../data/bayesian_annotator_combination/output/good_results/ner_al/'
    output_dir = './documents/figures/NER_AL/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    plot_active_learning_results(results_dir, output_dir)

