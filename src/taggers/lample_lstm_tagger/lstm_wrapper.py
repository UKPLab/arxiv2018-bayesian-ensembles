import os
import pickle
import sys
import time
from collections import OrderedDict

import sklearn.metrics as skm

from taggers.lample_lstm_tagger.utils import create_input
from taggers.lample_lstm_tagger.utils import models_path
from taggers.lample_lstm_tagger.loader import word_mapping, char_mapping
from taggers.lample_lstm_tagger.loader import prepare_dataset
from taggers.lample_lstm_tagger.model import Model

import numpy as np
from scipy.special import logsumexp

MAX_NO_EPOCHS = 25  # number of epochs recommended to try before testing against dev set for the first time


def data_to_lstm_format(text, doc_start, labels, nclasses=0, include_missing=False):

    # use only the data points with labels
    text = text[labels != -1]
    doc_start = doc_start[labels != -1]
    labels = labels[labels != -1]
    nannotations = len(doc_start)

    sentences_list = np.empty(int(np.sum(doc_start)), dtype=object)

    labels = labels.flatten()

    if include_missing:
        IOB_label = {1: 'O', 0: 'I-0', -1: 0}
        IOB_map = {'O': 1, 'I-0': 0, 0:-1}
    else:
        IOB_label = {1: 'O', 0: 'I-0'}
        IOB_map = {'O': 1, 'I-0': 0}

    label_values = np.unique(labels) if nclasses==0 else np.arange(nclasses, dtype=int)

    for val in label_values:
        if val <= 1:
            continue

        IOB_label[val] = 'I' if np.mod(val, 2) == 1 else 'B'
        IOB_label[val] = IOB_label[val] + '-%i' % ((val - 1) / 2)

        IOB_map[IOB_label[val]] = val

    docidx = -1
    for i in range(nannotations):

        label = labels[i]

        if doc_start[i]:
            docidx += 1

            sentence_list = []
            sentence_list.append([text[i][0], IOB_label[label]])
            sentences_list[docidx] = sentence_list
        else:
            sentence_list.append([text[i][0], IOB_label[label]])

    return sentences_list, IOB_map, IOB_label

def split_train_to_dev(gold_sentences):
    ndocs = gold_sentences.shape[0]
    devidxs = np.random.randint(0, ndocs, int(np.floor(ndocs * 0.5)))
    trainidxs = np.ones(len(gold_sentences), dtype=bool)
    trainidxs[devidxs] = 0
    train_sentences = gold_sentences[trainidxs]

    if len(devidxs) == 0:
        dev_sentences = gold_sentences
    else:
        dev_sentences = gold_sentences[devidxs]

    dev_gold = []
    for sen in dev_sentences:
        for tok in sen:
            dev_gold.append(tok[1])

    return train_sentences, dev_sentences, dev_gold

class LSTMWrapper(object):

    def __init__(self, models_path_alt=None, embeddings=None, reload_from_disk=False):
        self.model = None
        self.best_model_saved = False # flag to tell us whether the best model on dev was saved to disk
        self.best_dev = -np.inf
        self.last_score = -np.inf

        if models_path is None:
            self.models_path = models_path
        else:
            self.models_path = models_path_alt

        self.embeddings = embeddings

        self.reload_from_disk = reload_from_disk

    def run_epoch(self, epoch, niter_no_imprv, best_dev, last_score, compute_dev=True):
        epoch_costs = []
        print("Starting epoch %i..." % epoch)

        for i, index in enumerate(np.random.permutation(len(self.train_data))):
            # loop over random permutations of the training data
            input = create_input(self.train_data[index], self.parameters, True, self.singletons)
            new_cost = self.f_train(*input)
            epoch_costs.append(new_cost)

            if i % 500 == 0 and i > 0 == 0:
                print("%i, cost average: %f" % (i, np.mean(epoch_costs[-50:])))

        if compute_dev:
            agg, probs = self.predict_LSTM(self.dev_sentences)
            print('Dev predictions include: ' + str(np.unique(agg)))

            dev_score = skm.f1_score(self.dev_tags, agg, average='macro')

            print("Score on dev: %.5f" % dev_score)
            if dev_score > best_dev:
                best_dev = dev_score
                print("New best score on dev.")
                print("Saving model to disk...")
                self.model.save()
                self.best_model_saved = True
                self.best_dev = best_dev
                niter_no_imprv = 0

            else:
                niter_no_imprv += 1

            self.last_score = last_score
        else:
            dev_score = last_score

        print("Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs)))

        return niter_no_imprv, best_dev, dev_score

    def load_LSTM(self, crf_probs, all_sentences, IOB_label, nclasses):

        # parameters
        parameters = OrderedDict()
        parameters['tag_scheme'] = 'iob'
        parameters['lower'] = 0
        parameters['zeros'] = 0
        parameters['char_dim'] = 25
        parameters['char_lstm_dim'] = 25
        parameters['char_bidirect'] = 1
        parameters['word_dim'] = 300 # 100 --> this is the old setting
        parameters['word_lstm_dim'] = 100
        parameters['word_bidirect'] = 1
        parameters['pre_emb'] = self.embeddings # '../../data/bayesian_sequence_combination/glove.840B.300d.txt'
        # Nguyen 2017, which we compare against, seems not to have used embeddings as they are not mentioned
        # and performance matches our performance without embeddings. It makes sense to include them so we can make
        # the performance as realistic as possible, i.e., the performance you get if you really try to do transfer
        # learning in this kind of context.
        parameters['all_emb'] = 0
        parameters['cap_dim'] = 0
        parameters['crf'] = 1
        parameters['crf_probs'] = crf_probs # output probability of most likely sequence or just sequence labels. Only
        # applies if using crf already.
        parameters['dropout'] = 0.25
        parameters['lr_method'] = "adam-lr_.001"#"sgd-lr_.005" # this works better than the original set up according
        # to Reimers and Gurevych 2017. We also found it improved performance, at least with a very small number of epochs.

        #lr_decay = 0.9 # not currently used


        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

        # Initialize model
        model = Model(parameters=parameters, models_path=self.models_path)
        print("Model location: %s" % model.model_path)

        if self.reload_from_disk:
            model.reload()

        self.id_to_tag = IOB_label

        dico_words, word_to_id, id_to_word = word_mapping(all_sentences, parameters['lower'])

        # Create a dictionary and a mapping for words / POS tags / tags
        dico_chars, char_to_id, id_to_char = char_mapping(all_sentences)

        if self.reload_from_disk:
            word_to_id, char_to_id, IOB_map = model.reload_mappings()
        else:
            # Save the mappings to disk
            print('Saving the mappings to disk...')
            model.save_mappings(id_to_word, id_to_char, IOB_label)

        # Build the model
        pickle_f_train = os.path.join(model.model_path, "ftrain.pkl")
        pickle_f_eval = os.path.join(model.model_path, "feval.pkl")

        sys.setrecursionlimit(40000)

        if self.reload_from_disk and os.path.exists(pickle_f_train) and os.path.exists(pickle_f_eval):
            with open(pickle_f_train, 'rb') as fh:
                f_train = pickle.load(fh)
            with open(pickle_f_eval, 'rb') as fh:
                f_eval = pickle.load(fh)
        else:
            f_train, f_eval = model.build(**parameters)

            with open(pickle_f_train, 'wb') as fh:
                pickle.dump(f_train, fh)

            with open(pickle_f_eval, 'wb') as fh:
                pickle.dump(f_eval, fh)

        self.model = model
        self.parameters = parameters
        self.f_train = f_train
        self.f_eval = f_eval
        self.nclasses = nclasses

        return word_to_id, dico_words, char_to_id

    def train_LSTM(self, all_sentences, train_sentences, dev_sentences, dev_labels, IOB_map, IOB_label, nclasses,
                   n_epochs=MAX_NO_EPOCHS, freq_eval=100, max_niter_no_imprv=3, crf_probs=False, best_dev=-np.inf):

        word_to_id, dico_words, char_to_id = self.load_LSTM(crf_probs, all_sentences, IOB_label, nclasses)

        # Data parameters
        # Index data
        train_data, _ = prepare_dataset(
            train_sentences, word_to_id, char_to_id, IOB_map, self.parameters['lower']
        )
        dev_data, dev_tags = prepare_dataset(
            dev_sentences, word_to_id, char_to_id, IOB_map, self.parameters['lower']
        )
        if dev_labels is None:
            dev_tags = np.array(dev_tags).flatten()
            dev_tags = np.array([IOB_map[tag] for tag in dev_tags])
        else:
            dev_tags = np.array(dev_labels).flatten()

        #
        # Train network
        #
        # add new words to the dictionary
        for k in dico_words:
            if k not in word_to_id:
                word_to_id[k] = len(word_to_id)

        singletons = set([word_to_id[k] for k, v in list(dico_words.items()) if v == 1])

        # evaluate on dev every freq_eval steps
        last_score = best_dev
        niter_no_imprv = 0 # for early stopping

        self.train_data = train_data
        self.singletons = singletons
        self.dev_sentences = dev_sentences
        self.dev_tags = dev_tags
        self.IOB_map = IOB_map

        self.best_model_saved = False # reset the flag so we don't load old models

        for epoch in range(n_epochs):
            niter_no_imprv, best_dev, last_score = self.run_epoch(epoch, niter_no_imprv, best_dev,
                                                        last_score, (((epoch+1) % freq_eval) == 0) and (epoch < n_epochs))

            if niter_no_imprv >= max_niter_no_imprv:
                print("- early stopping %i epochs without improvement" % niter_no_imprv)
                break

        if self.best_model_saved:
            self.model.reload()

        return self.model, self.f_eval, best_dev

    def predict_LSTM(self, test_sentences):
        # Load existing model
        parameters = self.model.parameters
        lower = parameters['lower']

        # Load reverse mappings
        word_to_id, char_to_id, tag_to_id = [
            {v: k for k, v in list(x.items())}
            for x in [self.model.id_to_word, self.model.id_to_char, self.model.id_to_tag]
        ]

        test_data, _ = prepare_dataset(
            test_sentences, word_to_id, char_to_id, tag_to_id, lower, include_tags=False
        )

        start = time.time()

        print('Tagging...')
        count = 0

        agg = []
        probs = np.empty((0, self.nclasses))  # np.zeros((agg.shape[0], self.num_classes))

        for test_data_sen in test_data:
            input = create_input(test_data_sen, parameters, False)

            # Decoding
            if parameters['crf'] and parameters['crf_probs']:
                probs_sen = self.f_eval(*input)[:-1]
                probs_sen = np.exp(probs_sen - logsumexp(probs_sen, axis=1)[:, None])
                probs_sen = probs_sen[:, :self.nclasses] # there seem to be some additional classes. I think they are
                # used by the CRF for transitions from the states before and after the sequence, so we skip them here.
                y_preds = probs_sen.argmax(axis=1)
            elif parameters['crf']:
                # CRF but outputs labels instead of log-probabilities
                y_preds = np.array(self.f_eval(*input))[1:-1]
            else:
                y_preds = np.array(self.f_eval(*input))

            if not parameters['crf_probs']:
                probs_sen = np.zeros((len(y_preds), self.nclasses))
                probs_sen[np.arange(len(y_preds)), y_preds] = 1

            agg.extend(y_preds)
            probs = np.concatenate((probs, probs_sen), axis=0)

            count += 1
            if count % 1000 == 0:
                print('Tagged %i lines' % count)

        print('---- %i lines tagged in %.4fs ----' % (count, time.time() - start))

        agg = np.array(agg)

        return agg, probs