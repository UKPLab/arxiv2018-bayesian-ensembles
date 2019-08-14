'''

Hypothesis 1: ensemble learning with BCC improves performance with smaller amounts of data.


*Plot* the F1 score with increasing amounts of training data for BCC. Tested methods:
(BASELINE x) majority vote;
(BASELINE y) best individual source model performance
on the target domain;
(BASELINE z) model trained on data from all source domains;
(a) a DNN model trained on the gold standard for the target domain (a ceiling rather than a baseline);
(b) IBCC over the models from source domains, tested on the target domain;
(c) Like (b) but with BSC-seq (note that BSC includes a linear model);

Other things we can consider including later if there is space:

(d) an ensemble containing the DNN from target domain and DNNs from source domains
(do other domain models provide complementary information? is a model trained on
just a few labels still useful, does it add noise, or does it get ignored?).
(e) a shallow linear model trained on the target domain (does a simpler model work better with very
small data?)
(f) linear model added to the ensemble.

'''
import os, pandas as pd, numpy as np

from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.metrics import f1_score
import json

from sklearn_crfsuite.estimator import CRF

from lample_lstm_tagger.lstm_wrapper import LSTMWrapper, data_to_lstm_format

class simple_crf:

    def get_features(self, feats, doc_start):
        X = []
        for tok in range(len(doc_start)):
            if doc_start[tok]:
                Xdoc = []
                X.append(Xdoc)

            Xdoc.append({
                'unigram': feats[tok]
            })
        return X

    def split_labels_by_doc(self, labels, doc_start):
        y = []
        for tok in range(len(doc_start)):
            if doc_start[tok]:
                ydoc = []
                y.append(ydoc)

            ydoc.append(str(labels[tok]))

        return y

    def train_crf(self, text, doc_start, labels):
        X = self.get_features(text, doc_start)

        crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X, self.split_labels_by_doc(labels, doc_start))

        self.crf = crf

    def pred_crf(self, text, doc_start):
        X = self.get_features(text, doc_start)
        preds = self.crf.predict(X)

        preds = np.concatenate(preds).astype(int)

        return preds


datadir = os.path.expanduser('~/data/bayesian_sequence_combination/data/famulus_TEd')

'''
Structure of data folder:

Top dir: files contain all available data.
Sub-directories: subsets of data for different cases or ...?

In each directory there are the same files.

Files with names ending in '.txt_ed' contain:
    column 0 = text
    last column = doc_start
    columns 1, 2, 3, 4 = individual annotations for various numbers of classes using the BIO notation
    column 5 = gold in BIO notation?
    column 6, 7, 8, 9 = individual annotations as numeric class IDs
        0 = I
        1 = O
        2 = B
        3 = I
        4 = B ...
    column 10 = gold as numeric class IDs
    
For computing evaluation metrics, we only need to use the dev or test splits (only use test once we have a working setup).
For training BSC or IBCC, we use the training splits. 

We divide the data into 'folds' where a fold is one sub-directory. 
For each fold, there is a particular set of classes.

'''

subsets = sorted(os.listdir(datadir))

trgold = {}
trtext = {}
trdocstart = {}

tegold = {}
tetext = {}
tedocstart = {}

degold = {}
detext = {}
dedocstart = {}

# For each fold, load data
domains = []

debug = True
reload = False

for subset in subsets:

    subsetdir = os.path.join(datadir, subset)

    if not os.path.isdir(subsetdir):
        continue
    else:
        domains.append(subset)


    data = pd.read_csv(os.path.join(subsetdir, 'eda_train_ES.tsv'), sep='\t', usecols=[0, 10, 11],
                       names=['text', 'gold', 'doc_start'])
    trgold[subset] = data['gold'].values
    trtext[subset] = data['text'].values
    trdocstart[subset] = data['doc_start'].values

    data = pd.read_csv(os.path.join(subsetdir, 'eda_dev_ES.tsv'), sep='\t', usecols=[0, 10, 11],
                       names=['text', 'gold', 'doc_start'])
    degold[subset] = data['gold'].values
    detext[subset] = data['text'].values
    dedocstart[subset] = data['doc_start'].values

    data = pd.read_csv(os.path.join(subsetdir, 'eda_test_ES.tsv'), sep='\t', usecols=[0, 10, 11],
                       names=['text', 'gold', 'doc_start'])
    tegold[subset] = data['gold'].values
    tetext[subset] = data['text'].values
    tedocstart[subset] = data['doc_start'].values

# For each subset, train a base classifier and predict labels for all other subsets
tmpdir = os.path.expanduser('~/data/bayesian_sequence_combination/output/tmp')
if not os.path.exists(tmpdir):
    os.mkdir(tmpdir)

# For each subset, train a base classifier and predict labels for all other subsets
resdir = os.path.expanduser('~/data/bayesian_sequence_combination/output/famulus_TEd_results')
if not os.path.exists(resdir):
    os.mkdir(resdir)

nclasses = 9

preds = {
    'MV':[],
    'baseline_z':[],
    'a':[],
} # the predictions, with the base labeller's name as key. Each item is a list, where each entry is a list of
# predictions for one test subset.

res = {
    'MV': [],
    'baseline_z': [],
    'a':[],
} # F1 scores, indexed by base labeller name. Each item is a list of scores for a labeller for each test domain.

predfile = os.path.join(resdir, 'preds.json')
resfile = os.path.join(resdir, 'res.json')

if os.path.exists(predfile) and reload:
    with open(predfile, 'r') as fh:
        preds = json.load(fh)

    with open(resfile, 'r') as fh:
        res = json.load(fh)

    for key in res:
        print('Average F1 score=%f for labeller %s' % (np.mean(res[key]), key))

    # method (a), in-domain performance (Ceiling, not a baseline)
    res_out = []
    for domain in domains:
        res_out.append(res[domain])

    print('Average F1 score=%f for out-of-domain performance' % np.mean(res_out))
    print('Average F1 score=%f for in-domain performance' % np.mean(res['a']))

else:
    for domain in domains:
        subsetdir = os.path.join(tmpdir, domain)

        if debug:
            labeller = simple_crf()
            labeller.train_crf(trtext[domain], trdocstart[domain], trgold[domain])
        else:
            labeller = LSTMWrapper(subsetdir)#, os.path.expanduser('~/data/bayesian_sequence_combination/cc.de.300.vec'))

            N = len(trgold[domain])
            train_sentences, IOB_map, IOB_label = data_to_lstm_format(N, trtext[domain], trdocstart[domain],
                                                                          trgold[domain].flatten(), nclasses)

            Nde = len(degold[domain])
            dev_sentences, _, _ = data_to_lstm_format(Nde, detext[domain], dedocstart[domain],
                                                                          degold[domain].flatten(), nclasses)

            all_sentences = np.concatenate((train_sentences, dev_sentences), axis=0)

            labeller.train_LSTM(all_sentences, train_sentences, dev_sentences, degold[domain], IOB_map, IOB_label, nclasses,
                       n_epochs=1, freq_eval=1, max_niter_no_imprv=3, crf_probs=False)

        # given the model trained on the current domain, test it on all subsets
        for tedomain in domains:
            if tedomain[0] == '.':
                continue

            Nte = len(tegold[tedomain])

            if debug:
                preds_s = labeller.pred_crf(tetext[tedomain], tedocstart[tedomain])
            else:
                test_sentences, _, _ = data_to_lstm_format(Nte, tetext[tedomain], tedocstart[tedomain], np.ones(Nte), nclasses)
                preds_s, _ = labeller.predict_LSTM(test_sentences)

            if domain not in preds:
                preds[domain] = []

            if tedomain == domain:
                # separate the in-domain predictions
                preds['a'].append(preds_s.tolist())
                preds[domain].append((np.ones(len(preds_s)) - 1).tolist()) # no predictions
            else:
                preds[domain].append(preds_s.tolist())

            res_s = f1_score(tegold[tedomain], preds_s, average='macro')

            if tedomain == domain:
                res['a'].append(res_s)
            else:
                if domain not in res:
                    res[domain] = []

                res[domain].append(res_s)

            print('F1 score=%f for base labeller trained on %s and tested on %s' % (res_s, domain, tedomain))

        print('Average F1 score=%f for base labeller trained on %s, tested on other domains' % (np.mean(res[domain]), domain))

    # method (a), in-domain performance (Ceiling, not a baseline)
    res_out = []
    for domain in domains:
        res_out.append(res[domain])

    print('Average F1 score=%f for out-of-domain performance' % np.mean(res_out))
    print('Average F1 score=%f for in-domain performance' % np.mean(res['a']))


with open(predfile, 'w') as fh:
    json.dump(preds, fh)

with open(resfile, 'w') as fh:
    json.dump(res, fh)

# BASELINE X -- majority vote from labellers trained on other domains
for didx, tedomain in enumerate(domains):
    votes = []
    for base in domains:
        votes.append(np.array(preds[base][didx])[:, None]) # predictions for model trained on base domain, tested on tedomain

    votes = np.concatenate(votes, axis=1) # put into matrix format
    counts = np.zeros((votes.shape[0], nclasses))
    for c in range(nclasses):
        counts[:, c] = np.sum(votes==c, axis=1)

    mv = np.argmax(counts, axis=1)

    preds['MV'].append(mv.tolist())

    res_s = f1_score(tegold[tedomain], mv, average='macro')
    res['MV'] = []
    res['MV'].append(res_s)

    print('F1 score=%f for majority vote tested on %s' % (res_s, tedomain))

print('Average F1 score=%f for majority vote' % np.mean(res['MV']))

with open(predfile, 'w') as fh:
    json.dump(preds, fh)

with open(resfile, 'w') as fh:
    json.dump(res, fh)

# BASELINE Z -- model trained on data from all source domains
if 'baseline_z' not in res or not len(res['baseline_z']):
    for tedomain in domains:

        subsetdir = os.path.join(tmpdir, tedomain + '_baseline_z')

        labeller = LSTMWrapper(subsetdir)#, os.path.expanduser('~/data/bayesian_sequence_combination/cc.de.300.vec'))

        trtext_allsources = []
        trdocstart_allsources = []
        trgold_allsources = []

        detext_allsources = []
        dedocstart_allsources = []
        degold_allsources = []

        for trsubset in domains:
            if trsubset[0] == '.':
                continue

            if tedomain == trsubset:
                continue  # only test on cross-domain

            trtext_allsources.append(trtext[trsubset].flatten())
            trdocstart_allsources.append(trdocstart[trsubset].flatten())
            trgold_allsources.append(trgold[trsubset].flatten())

            detext_allsources.append(detext[trsubset].flatten())
            dedocstart_allsources.append(dedocstart[trsubset].flatten())
            degold_allsources.append(degold[trsubset].flatten())

        trtext_allsources = np.concatenate(trtext_allsources)
        trdocstart_allsources = np.concatenate(trdocstart_allsources)
        trgold_allsources = np.concatenate(trgold_allsources)

        detext_allsources = np.concatenate(detext_allsources)
        dedocstart_allsources = np.concatenate(dedocstart_allsources)
        degold_allsources = np.concatenate(degold_allsources)

        if debug:
            labeller = simple_crf()
            labeller.train_crf(trtext_allsources, trdocstart_allsources, trgold_allsources)

            preds_s = labeller.pred_crf(tetext[tedomain], tedocstart[tedomain])
        else:
            N = len(trgold_allsources)

            train_sentences, IOB_map, IOB_label = data_to_lstm_format(N, trtext_allsources, trdocstart_allsources,
                                                                          trgold_allsources, nclasses)

            Nde = len(degold_allsources)

            dev_sentences, _, _ = data_to_lstm_format(Nde, detext_allsources, dedocstart_allsources,
                                                      degold_allsources, nclasses)

            all_sentences = np.concatenate((train_sentences, dev_sentences), axis=0)

            labeller.train_LSTM(all_sentences, train_sentences, dev_sentences, degold_allsources, IOB_map, IOB_label, nclasses,
                       n_epochs=100, freq_eval=1, max_niter_no_imprv=5, crf_probs=False)

            # given the model trained on the all domains but one, test it on the held-out domain
            Nte = len(tegold[tedomain])

            test_sentences, _, _ = data_to_lstm_format(Nte, tetext[tedomain], tedocstart[tedomain], np.ones(Nte), nclasses)

            preds_s, _ = labeller.predict_LSTM(test_sentences)

        preds['baseline_z'].append(preds_s.tolist())

        res_s = f1_score(tegold[tedomain], preds_s, average='macro')

        res['baseline_z'] = []

        res['baseline_z'].append(res_s)

        print('F1 score=%f for leave-one-out training, tested on %s' % (res_s, tedomain))

    print('Average F1 score=%f for baseline_z trained with leave-one-out' % np.mean(res['baseline_z']))

    with open(predfile, 'w') as fh:
        json.dump(preds, fh)

    with open(resfile, 'w') as fh:
        json.dump(res, fh)