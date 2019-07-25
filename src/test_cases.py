'''

Hypothesis 1: ensemble learning with BCC improves performance with smaller amounts of data.

Plot the F1 score with increasing amounts of training data for BCC. Tested methods:
(a) a DNN model trained on the target domain (a target rather than a baseline) using only
the available data;
(b) IBCC over the models from source domains, tested on the target domain;
(c) Like (b) but with BSC-seq;
(d) an ensemble containing the DNN from target domain and DNNs from source domains
(do other domain models provide complementary information? is a model trained on
just a few labels still useful, does it add noise, or does it get ignored?).
(e) a shallow linear model trained on the target domain (does a simpler model work better with very
small data?)
(f) linear model added to the ensemble.

'''
import os, pandas as pd, numpy as np
from conllu import parse, parse_incr
from sklearn.metrics import f1_score

datadir = os.path.expanduser('~/data/bayesian_sequence_combination/data/famulus')

'''
Structure of data folder:

Top dir: files contain all available data.
Sub-directories: subsets of data for different cases or ...?

In each directory there are the same files.

seq_train, seq_dev, seq_test contain the labels for all 9 classes.
    blank lines separate each document.


Files with names ending in '_ed.csv' contain:
    column 0 = text
    column 1 = doc_start
    column 2: = predictions for various numbers of classes, whose labels are given in files ending in '_ed_labels.list'.
    
Files ending in '_ed_all.csv' are the same as '_ed.csv' except they contain predictions for all subsets for the model 
trained on a particular subset.

These files contain either the predictions for the train, test or dev splits. 

For computing evaluation metrics, we only need to use the dev or test splits (only use test once we have a working setup).
For training BSC or IBCC, we use the training splits. 

We divide the data into 'folds' where a fold is a data subset + a condition. For each fold, there is a particular
set of classes. We take all the predictions from models trained on other subsets and other conditions as base 
classifiers for our model. 

'''

subsets = sorted(os.listdir(datadir))

# For each fold, load predictions from the target model for the test split
for subset in subsets:

    subset = os.path.join(datadir, subset)

    if not os.path.isdir(subset):
        continue

    # get the subtaks for this task
    filenames = os.listdir(subset)

    if len(filenames) <= 3: # it is missing the model predictions
        continue

    # tasks = []
    # for filename in filenames:
    #     if not '_test_ed.csv' in filename:
    #         continue
    #
    #     tasks.append(filename[:-11])

    tasks = [
        'erg_sonstiges',
        'erg_schreiben',
        'erg_lesen',
        'erg_leistung',
        'erg_emotion',
        'dia_sozial',
        'dia_intelligenz',
        'dia_autismus',
        'dia_ADHS'
    ]


    class_names = []

    for j, task in enumerate(tasks):

        class_names.append(np.genfromtxt(subset + '/' + task + '_test_ed_labels.list', dtype=str))

    # Load the gold standard
    # with open(subset + '/seq_test.txt', 'r', encoding="utf-8") as fh:
    #     gold_data = parse_incr(fh)
    #
    #     t_gold = []
    #     for sentence in gold_data:
    #
    #         for tok in sentence:
    #             t_gold_i = []
    #             for j in range(len(tok)):
    #                 t_gold_i.append(np.argwhere(class_names[j] == tok[j])[0][0])
    #
    #             t_gold.append(t_gold_i)
    #
    # t_gold = np.array(t_gold)

    goldfile = os.path.join(subset, 'seq_test.txt')
    gold_data = pd.read_csv(goldfile, sep='\t', header=None)
    gold_data = gold_data.iloc[:, 1:]
    for j in range(gold_data.shape[1]):
        for k, class_name in enumerate(class_names[j]):
            gold_data.iloc[np.argwhere(gold_data.iloc[:, j] == class_name).flatten(), j] = k

    t_gold = gold_data.values

    C = None

    print('loading the predictions and computing performance on the source tasks')

    # load predictions
    for j, task in enumerate(tasks):

        # Load test predictions for the target model
        target_preds = pd.read_csv(os.path.join(subset, task + '_test_ed.csv'), header=None, sep='\t')
        target_preds = target_preds.dropna(axis='columns', how='all')

        target_text = target_preds[0]
        target_starts = target_preds[1].values.astype(int)

        target_preds = target_preds.iloc[:, 2:].values.astype(float)

        # we try taking the most likely labels for now
        target_preds = np.argmax(target_preds, axis=1)

        if C is None:
            C = target_preds[:, None]
        else:
            C = np.concatenate((C, target_preds[:, None]), axis=1)

        # Compute F1 score
        f1 = f1_score(t_gold[:, j], target_preds, average='macro')

        print(f1)