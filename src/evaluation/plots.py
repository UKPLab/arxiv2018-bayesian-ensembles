'''
Created on Nov 1, 2016

@author: Melvin Laux
'''

#import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

PARAM_NAMES = ['acc_bias',
               'miss_bias',
               'short_bias',
               'num_docs',
               'doc_length',
               'group_sizes',
               'no. labels'
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

def nice_names(methods):

    nicer = {
        'bac_acc': 'BSC-acc',
        'bac_vec': 'BSC-CV',
        'bac_seq': 'BSC-seq',
        'bac_ibcc': 'BSC-CM',
        'bac_mace': 'BSC-MACE',
        'HMM_crowd': 'HMMcrowd',
        'ibcc': 'IBCC',
        'mace': 'MACE',
        'ds': 'DS',
        'majority': 'MV'
    }

    names = []
    for method in methods:
        if method in nicer:
            names.append(nicer[method])
        else:
            names.append(method)

    return names

def make_plot(methods, param_idx, x_vals, y_vals, x_ticks_labels, ylabel, title=None):

    styles = ['-', '--', '-.', ':']
    markers = ['o', 'v', 's', 'p', '*']

    matplotlib.rcParams.update({'font.size': 12})

    for j in range(len(methods)):
        plt.plot(x_vals, np.mean(y_vals[:, j, :], 1), label=methods[j], ls=styles[j%4], marker=markers[j%5])

    plt.legend(loc='lower right') #loc='best')

    if title is not None:
        plt.title(title)

    plt.ylabel(ylabel)
    plt.xlabel(PARAM_NAMES[param_idx])
    #plt.xticks(x_vals, x_ticks_labels)
    plt.grid(True)
    plt.grid(True, which='Minor', axis='y')

    if np.min(y_vals) < 0:
        plt.ylim([np.min(y_vals), np.max([1, np.max(y_vals)])])
    #elif not np.isinf(np.max(y_vals)) and not np.isnan(np.max(y_vals)):
    #    plt.ylim([0, np.max([1, np.max(y_vals)])])

    #plt.tight_layout()


def plot_results(param_values, methods, param_idx, results, show_plot=False, save_plot=False, output_dir='/output/',
                 score_names=None, title=None):
    # Plots the results for varying the parameter specified by param_idx

    methods = nice_names(methods)

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
            plt.savefig(output_dir + 'plot_' + score_names[i] + '.pdf')
            plt.clf()

        if show_plot:
            plt.show()

        if i == 5:
            make_plot(methods, param_idx, x_vals, results[:,i,:,:], x_ticks_labels, score_names[i], title)
            plt.ylim([0,1])

            if save_plot:
                print('Saving plot...')
                plt.savefig(output_dir + 'plot_' + score_names[i] + '_zoomed' + '.pdf')
                plt.clf()

            if show_plot:
                plt.show()

def plot_active_learning_results(results_dir, output_dir, intervals, result_str='result_'):

    # old setups
    #ndocs = np.array([605, 1210, 1815, 2420, 3025, 3630, 4235, 4840, 5445, 6045]) # NER dataset
    #ndocs = np.array([606, 1212, 1818, 2424, 3030, 3636, 4242, 4848, 5454, 6056])
    #ndocs = np.array([929, 1858, 2787])

    # for NER
    if intervals == 'NER':
        ndocs = np.array([1486, 2972, 4458, 5944, 7430, 8916, 10402, 11888, 13374, 14860])
    elif intervals == 'PICO':
        # for PICO
        ndocs = np.array([2843, 5686, 8529, 11372, 14215, 17058, 19901, 22744, 25587, 28430])
    elif intervals == 'PICOsmall':
        ndocs = np.array([384, 764, 1132, 1488, 1834, 2132, 2300])

    methods = np.array([
        'majority',
        'HMM_crowd',
        'HMM_crowd_then_LSTM',
        'bac_ibcc_integrateBOF',
        'bac_seq_integrateBOF',
        'bac_seq_integrateBOF_then_LSTM',
        'bac_seq_integrateBOF_integrateLSTM_atEnd',
    ])

    method_names = np.array([
        'MV',
        'HMMcrowd',
        'HMMcrowd->LSTM',
        'BSC-CM',
        'BSC-seq',
        'BSC-seq->LSTM',
        'BSC-seq+LSTM',
    ])


    plot_methods = np.zeros(len(methods), dtype=bool)

    # results: y-value, metric, methods, runs
    results = np.zeros((len(ndocs), len(SCORE_NAMES), len(methods), 1))
    results_nocrowd = np.zeros((len(ndocs), len(SCORE_NAMES), len(methods), 1))

    run_counts = np.zeros((len(ndocs), len(SCORE_NAMES), len(methods), 1))
    run_counts_nocrowd = np.zeros((len(ndocs), len(SCORE_NAMES), len(methods), 1))

    resfiles = os.listdir(results_dir)

    for resfile in resfiles:

        if result_str not in resfile:
            continue

        if result_str + 'std_started' in resfile:
            continue
        elif result_str + 'std_nocrowd' in resfile:
            continue

        ndocsidx = int(resfile.split('.csv')[0].split('Nseen')[-1])
        print(ndocsidx)
        if ndocsidx not in ndocs:
            print('skipping this value of ndocs=%i' % ndocsidx)
            continue

        ndocsidx = np.argwhere(ndocs == ndocsidx)[0][0]

        if resfile.split('.')[-1] == 'csv':
            res = pd.read_csv(os.path.join(results_dir, resfile))

            for col in res.columns:
                thismethod = col.strip("\\# '")
                print('plotting method "%s"' % thismethod)

                if thismethod in methods:
                    methodidx = np.argwhere(methods == thismethod)[0][0]
                else:
                    methodidx = len(methods)
                    methods = methods.append(thismethod)

                plot_methods[methodidx] = True

                if result_str + 'started' in resfile:
                    results[ndocsidx, :, methodidx, 0] = (res[col] + run_counts[ndocsidx, :, methodidx, 0] * \
                        results[ndocsidx, :, methodidx, 0]) / (run_counts[ndocsidx, :, methodidx, 0] + 1)
                    run_counts[ndocsidx, :, methodidx, 0] += 1
                elif result_str + 'nocrowd_started' in resfile:
                    results_nocrowd[ndocsidx, :, methodidx, 0] = (res[col] + run_counts_nocrowd[
                        ndocsidx, :, methodidx, 0] * results_nocrowd[ndocsidx, :, methodidx, 0]) / (
                            run_counts_nocrowd[ndocsidx, :, methodidx, 0] + 1)
                    run_counts_nocrowd[ndocsidx, :, methodidx, 0] += 1

        elif resfile.split('.')[-1] == 'tex':
            res = pd.read_csv(os.path.join(results_dir, resfile), sep='&', header=None)

            for row in range(res.shape[0]):
                methodidx = np.argwhere(methods == res[res.columns[0]][row].strip())[0][0]

                plot_methods[methodidx] = True

                recomputed_order = [6, 7, 8, 3, 4, 5, 13]

                if result_str + 'started' in resfile:
                    for m, metric in enumerate(recomputed_order):
                        results[ndocsidx, metric, methodidx, 0] = res[res.columns[m+1]][row]
                elif result_str + 'nocrowd_started' in resfile:
                    for m, metric in enumerate(recomputed_order):
                        results_nocrowd[ndocsidx, metric, methodidx, 0] = res[res.columns[m+1]][row]

        else:
            continue

    output_pool_dir = os.path.join(output_dir, 'pool/')
    output_test_dir = os.path.join(output_dir, 'test/')

    plot_results(ndocs, method_names[plot_methods], 6, results[:, :, plot_methods, :], False, True, output_pool_dir, SCORE_NAMES)#,
                 #title='Active Learning: Pool Data')
    plot_results(ndocs, method_names[plot_methods], 6, results_nocrowd[:, :, plot_methods, :], False, True, output_test_dir, SCORE_NAMES)#,
                 #title='Active Learning: Test Data')

    print('Counts of runs with results on pool data:')
    print(run_counts[0, 0, :, 0])
    print('Counts of runs with results on test data:')
    print(run_counts_nocrowd[0, 0, :, 0])

