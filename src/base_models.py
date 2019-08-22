import os, numpy as np, json

from helpers import evaluate
from seq_taggers import simple_crf, lample #, flair_ner, flair_pos

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


def run_base_models(dataset, spantype, base_model_str='flair', reload=True, verbose=False):
    '''
    Use this to run the base sequence taggers and save their predictions.

    :param base_model_str: one of 'flair', 'flair-pos',  'flair-ner',  'crf', 'bilstm-crf'
    :param reload: if set to false, it will not overwrite existing results, otherwise all models are rerun
    :return:
    '''
    if verbose:
        print('BASE: Using base model %s' % base_model_str)

    # For each subset, train a base classifier and predict labels for all other subsets
    tmpdir = os.path.expanduser('~/data/bayesian_sequence_combination/output/tmp')
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    # For each subset, train a base classifier and predict labels for all other subsets
    resdir = os.path.expanduser('~/data/bayesian_sequence_combination/output/famulus_TEd_results_%s_spantype%i'
                                % (base_model_str, spantype))
    if not os.path.exists(resdir):
        os.mkdir(resdir)

    preds = {
        'a': [],
    }  # the predictions, with the base labeller's name as key. Each item is a list, where each entry is a list of
    # predictions for one test subset.

    trpreds = {
    }  # predictions on the training sets -- need this for training our aggregation methods

    res = {
        'a': [],
    }  # F1 scores, indexed by base labeller name. Each item is a list of scores for a labeller for each test domain.

    predfile = os.path.join(resdir, 'preds.json')
    trpredfile = os.path.join(resdir, 'trpreds.json')
    resfile = os.path.join(resdir, 'res.json')

    np.random.seed(23409809)

    # BASELINE Z -- model trained on data from all source domains
    labeller_every = None
    if 'baseline_every' not in res or not len(res['baseline_every']):

        if base_model_str == 'crf':
            labeller_every = simple_crf()
        elif base_model_str == 'bilstm-crf':
            labeller_every = lample(os.path.join(tmpdir, 'baseline_every'), 3)
        else:
            print('BASE: Not running pretrained models again')

    if labeller_every is not None:

        res['baseline_z'] = []
        res['baseline_every'] = []

        trpreds['baseline_z'] = []
        trpreds['baseline_every'] = []

        preds['baseline_z'] = []
        preds['baseline_every'] = []


        trtext_every = []
        trdocstart_every = []
        trgold_every = []

        detext_every = []
        dedocstart_every = []
        degold_every = []

        for trsubset in dataset.domains:
            trtext_every.append(dataset.trtext[trsubset].flatten())
            trdocstart_every.append(dataset.trdocstart[trsubset].flatten())
            trgold_every.append(dataset.trgold[trsubset].flatten())

            detext_every.append(dataset.detext[trsubset].flatten())
            dedocstart_every.append(dataset.dedocstart[trsubset].flatten())
            degold_every.append(dataset.degold[trsubset].flatten())

        trtext_every = np.concatenate(trtext_every)
        trdocstart_every = np.concatenate(trdocstart_every)
        trgold_every = np.concatenate(trgold_every)

        detext_every = np.concatenate(detext_every)
        dedocstart_every = np.concatenate(dedocstart_every)
        degold_every = np.concatenate(degold_every)

        print('BASE: The dataset contains: %i tokens and %i docs' % (len(trtext_every), np.sum(trdocstart_every)))

        labeller_every.train(trtext_every, trdocstart_every, trgold_every, detext_every, dedocstart_every, degold_every)

        for tedomain in dataset.domains:

            trtext_allsources = []
            trdocstart_allsources = []
            trgold_allsources = []

            detext_allsources = []
            dedocstart_allsources = []
            degold_allsources = []

            for trsubset in dataset.domains:
                if trsubset[0] == '.':
                    continue

                if tedomain == trsubset:
                    continue  # only test on cross-domain

                trtext_allsources.append(dataset.trtext[trsubset].flatten())
                trdocstart_allsources.append(dataset.trdocstart[trsubset].flatten())
                trgold_allsources.append(dataset.trgold[trsubset].flatten())

                detext_allsources.append(dataset.detext[trsubset].flatten())
                dedocstart_allsources.append(dataset.dedocstart[trsubset].flatten())
                degold_allsources.append(dataset.degold[trsubset].flatten())

            trtext_allsources = np.concatenate(trtext_allsources)
            trdocstart_allsources = np.concatenate(trdocstart_allsources)
            trgold_allsources = np.concatenate(trgold_allsources)

            detext_allsources = np.concatenate(detext_allsources)
            dedocstart_allsources = np.concatenate(dedocstart_allsources)
            degold_allsources = np.concatenate(degold_allsources)

            if base_model_str == 'crf':
                labeller = simple_crf()
            elif base_model_str == 'bilstm-crf':
                labeller = lample(os.path.join(tmpdir, os.path.join(tmpdir, tedomain + '_baseline_z')), nclasses=3)

            labeller.train(trtext_allsources, trdocstart_allsources, trgold_allsources, detext_allsources,
                           dedocstart_allsources, degold_allsources)

            preds_s = labeller.predict(dataset.tetext[tedomain], dataset.tedocstart[tedomain])
            trpreds_s = labeller.predict(dataset.trtext[tedomain], dataset.trdocstart[tedomain])

            preds_s_every = labeller_every.predict(dataset.tetext[tedomain], dataset.tedocstart[tedomain])

            preds['baseline_z'].append(preds_s.tolist())
            trpreds['baseline_z'].append(trpreds_s.tolist())

            preds['baseline_every'].append(preds_s_every.tolist())

            res_s = evaluate(preds_s, dataset.tegold[tedomain], dataset.tedocstart[tedomain])
            res['baseline_z'].append(res_s)

            res_every = evaluate(preds_s_every, dataset.tegold[tedomain], dataset.tedocstart[tedomain])
            res['baseline_every'].append(res_every)

            if verbose:
                print('BASE: F1 score=%f for leave-one-out training, tested on %s' % (res_s, tedomain))
                print('BASE: F1 score=%f for model trained on all, tested on %s' % (res_every, tedomain))

        print('BASE: Average F1 score=%f for baseline_z trained with leave-one-out' % np.mean(res['baseline_z']))
        print('BASE: Average F1 score=%f for baseline_every, which is trained on all' % np.mean(res['baseline_every']))

        with open(predfile, 'w') as fh:
            json.dump(preds, fh)

        with open(trpredfile, 'w') as fh:
            json.dump(trpreds, fh)

        with open(resfile, 'w') as fh:
            json.dump(res, fh)

    if os.path.exists(predfile) and reload:
        with open(predfile, 'r') as fh:
            preds = json.load(fh)

        with open(trpredfile, 'r') as fh:
            trpreds = json.load(fh)

        with open(resfile, 'r') as fh:
            res = json.load(fh)

        if verbose:
            for key in res:
                print('BASE: Average F1 score=%f for labeller %s' % (np.mean(res[key]), key))

        # method (a), in-domain performance (Ceiling, not a baseline)
        res_out = []
        for domain in dataset.domains:
            res_out.append(res[domain])

        print('BASE: Average F1 score=%f for out-of-domain performance' % np.mean(res_out))
        print('BASE: Average F1 score=%f for in-domain performance' % np.mean(res['a']))

    else:
        for domain in dataset.domains:
            subsetdir = os.path.join(tmpdir, domain)

            if base_model_str == 'crf':
                labeller = simple_crf()
            elif base_model_str == 'bilstm-crf':
                labeller = lample(subsetdir, 3)
            # elif base_model_str == 'flair-ner':
            #     labeller = flair_ner()
            # elif base_model_str == 'flair-pos':
            #     labeller = flair_pos()
            # elif base_model_str == 'flair':
            #     labeller = flair()

            labeller.train(dataset.trtext[domain], dataset.trdocstart[domain], dataset.trgold[domain],
                           dataset.detext[domain], dataset.dedocstart[domain], dataset.degold[domain])

            # given the model trained on the current domain, test it on all subsets
            for tedomain in dataset.domains:
                if tedomain[0] == '.':
                    continue

                preds_s = labeller.predict(dataset.tetext[tedomain], dataset.tedocstart[tedomain])
                trpreds_s = labeller.predict(dataset.trtext[tedomain], dataset.trdocstart[tedomain])

                if domain not in preds:
                    preds[domain] = []
                    trpreds[domain] = []

                if tedomain == domain:
                    # separate the in-domain predictions
                    preds['a'].append(preds_s.tolist())
                    preds[domain].append((np.zeros(len(preds_s)) - 1).tolist())  # no predictions
                    trpreds[domain].append((np.zeros(len(preds_s)) - 1).tolist())  # no predictions
                else:
                    preds[domain].append(preds_s.tolist())
                    trpreds[domain].append(trpreds_s.tolist())

                res_s = evaluate(preds_s, dataset.tegold[tedomain], dataset.tedocstart[tedomain], f1type='all')

                if tedomain == domain:
                    res['a'].append(res_s)
                else:
                    if domain not in res:
                        res[domain] = []

                    res[domain].append(res_s)
                if verbose:
                    print('BASE: F1 score=%f for base labeller trained on %s and tested on %s' % (res_s, domain, tedomain))
            if verbose:
                print('BASE: Average F1 score=%f for base labeller trained on %s, tested on other domains' % (
                    np.mean(res[domain]), domain))

        # method (a), in-domain performance (Ceiling, not a baseline)
        res_out = []
        for domain in dataset.domains:
            res_out.append(res[domain])

        print('BASE: Average F1 score=%f for out-of-domain performance' % np.mean(res_out))
        print('BASE: Average F1 score=%f for in-domain performance' % np.mean(res['a']))

        with open(predfile, 'w') as fh:
            json.dump(preds, fh)

        with open(trpredfile, 'w') as fh:
            json.dump(trpreds, fh)

        with open(resfile, 'w') as fh:
            json.dump(res, fh)

    return preds, trpreds, res
