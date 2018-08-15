'''

Use the csv files in a given directory to recompute the performance metrics.

Output a snippet of latex containing all the metrics, including the posterior and span-related metrics.

'''
import pickle
from argparse import ArgumentParser

import data.load_data as load_data
from evaluation.experiment import calculate_scores
import pandas as pd
import numpy as np
import os


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("dir", help="Directory containing the pred_xxx.csv files for recomputing performance.")
    parser.add_argument("dataset", help="use NER or PICO to select on of the two loaders for the gold labels")
    parser.add_argument("strict", help="STRICT=use strict span-level precision, recall and f1, or RELAXED=count fractions of span matches")
    args = parser.parse_args()

    if args.dataset == 'NER':
        gt, annos, doc_start, _, gt_nocrowd, doc_start_nocrowd, _, _, gt_val, doc_start_val, _ = \
            load_data.load_ner_data(False)



    elif args.dataset == 'PICO': # task 1
        gt, annos, doc_start, _, _, gt_nocrowd, doc_start_nocrowd, _ = load_data.load_biomedical_data(False)

    elif args.dataset == 'PICO2': # task 2
        gt, annos, doc_start, _, _, _, _, _ = load_data.load_biomedical_data(False)

        gold_labelled = gt.flatten() != -1

        gt_nocrowd = gt[gold_labelled]
        doc_start_nocrowd = doc_start[gold_labelled]
    else:
        print('Invalid dataset %s' % args.dataset)

    nclasses = np.max(gt) + 1

    if not os.path.isdir(args.dir):
        print('The dir argument must specify a directory.')
    else:
        for fname in os.listdir(args.dir):

            if "pred_" not in fname or '.csv' not in fname:
                continue

            print('Processing file %s' % fname)
            pred_path = os.path.join(args.dir, fname)
            agg = pd.read_csv(pred_path)

            # use the pred files to get the timestamps from the name. We need this to load the matching probabilities
            # and output a file with same timestamp.
            if 'pred_nocrowd_' in fname:
                timestamp = fname[len('pred_nocrowd_'):]
                probfile = 'probs_nocrowd_' + timestamp[:-len('.csv')] + '.pkl'
                outfile = 'recomputed_nocrowd_' + timestamp + '.tex'

                gt_this_task = gt_nocrowd
                doc_start_this_task = doc_start_nocrowd
            else:
                timestamp = fname[len('pred_'):]
                probfile = 'probs_' + timestamp[:-len('.csv')] + '.pkl'
                outfile = 'recomputed_' + timestamp + '.tex'

                gt_this_task = gt
                doc_start_this_task = doc_start

            outfile = os.path.join(args.dir, outfile)
            probfile = os.path.join(args.dir, probfile)

            with open(probfile, 'rb') as fh:
                probs = pickle.load(fh)

            result_to_print = ''

            for m, method in enumerate(agg.columns):

                print(agg[method].values.shape)

                result, std_result = calculate_scores(nclasses,
                                                      False,
                                                      agg[method].values,
                                                      gt_this_task.flatten().astype(int),
                                                      probs[:, :, m],
                                                      doc_start_this_task,
                                                      bootstrapping=True,
                                                      print_per_class_results=True)

                result_to_print += method.strip("# '") + ' & '

                # Output order:
                if args.strict == 'STRICT':
                    # prec, strict spans
                    result_to_print += '%.2f & ' % (result[6] * 100)

                    # rec, strict spans
                    result_to_print += '%.2f & ' % (result[7] * 100)

                    # f1, strict spans
                    result_to_print += '%.2f & ' % (result[8] * 100)

                    # f1 std, strict spans
                    if std_result is not None:
                        result_to_print += '%.2f & ' % (std_result[8] * 100)

                elif args.strict == 'RELAXED':
                    # prec, relaxed spans
                    result_to_print += '%.2f & ' % (result[9] * 100)

                    # rec, relaxed spans
                    result_to_print += '%.2f & ' % (result[10] * 100)

                    # f1, relaxed spans
                    result_to_print += '%.2f & ' % (result[11] * 100)

                    # f1 std, relaxed spans
                    if std_result is not None:
                        result_to_print += '%.2f & ' % (std_result[11] * 100)

                # f1, token level
                result_to_print += '%.2f & ' % (result[3] * 100)

                # f1 std, token level
                if std_result is not None:
                    result_to_print += '%.2f & ' % (std_result[3] * 100)

                # auc
                result_to_print += '%.3f & ' % result[4]

                # cee
                result_to_print += '%.2f & ' % result[5]

                # n invalid label transitions
                result_to_print += '%i & ' % result[13] #* gt_this_task.shape[0])

                result_to_print += '\\nonumber \\\\ \n'


            print(result_to_print)
            with open(outfile, 'w') as fh:
                fh.write(result_to_print)
