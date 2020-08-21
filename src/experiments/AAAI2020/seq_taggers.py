'''
Contains simple iterfaces to sequence taggers used for testing the ensembling methods.
'''
import os
import random

import numpy as np
#from flair.data import Sentence
#from flair.models import SequenceTagger
from sklearn_crfsuite.estimator import CRF

from experiments.AAAI2020.helpers import get_root_dir
from taggers.lample_lstm_tagger.lstm_wrapper import LSTMWrapper, data_to_lstm_format


embpath = os.path.join(get_root_dir(), 'cc.de.300.vec')


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

    def train(self, text, doc_start, labels, detext=None, dedoc_start=None, delabels=None):
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

    def predict(self, text, doc_start):
        X = self.get_features(text, doc_start)
        preds = self.crf.predict(X)

        preds = np.concatenate(preds).astype(int)

        return preds


def convert_str_tag_to_num(tag, types):
    tag = tag.value

    if tag == 'O' or len(tag) == 0:
        num_tag = 1
        return num_tag, types

    print(tag)
    typestr = tag.split('-')[1]

    if typestr not in types:
        types.append(typestr)

    typeidx = np.argwhere(np.array(types) == typestr).flatten()[0]

    if 'B-' in tag or 'S-' in tag:
        num_tag = (typeidx + 1) * 2
    elif 'I-' in tag or 'E-' in tag:
        if typeidx == 0:
            num_tag = 0
        else:
            num_tag = (typeidx + 1) * 2 - 1
    else:
        print('what is this? ' + tag)

    return num_tag, types


# class flair:
#
#     def train(self, model_str):
#         self.tagger = SequenceTagger.load(model_str)
#         # this is a pre-trained model, so do no more training here
#
#
#     def predict(self, text, doc_start):
#
#         docs = np.split(text, np.argwhere(doc_start).flatten())[1:]
#
#         docs = [Sentence(" ".join(doc), use_tokenizer=False) for doc in docs]
#         docs = self.tagger.predict(docs)
#
#         preds = []
#         types = []
#
#         for doc in docs:
#             for tok in doc:
#                 #print(tok)
#                 tag = tok.get_tag('ner')
#                 #print(tag)
#
#                 num_tag, types = convert_str_tag_to_num(tag, types)
#                 preds.append(num_tag)
#
#         return np.array(preds)
#
#
# class flair_ner(flair):
#     def train(self, text, doc_start, labels, detext=None, dedoc_start=None, delabels=None):
#         super(flair_ner, self).train('de-pos')
#
#
# class flair_pos(flair):
#
#     def train(self, text, doc_start, labels, detext=None, dedoc_start=None, delabels=None):
#         super(flair_pos, self).train('de-pos')
#         # this is a pre-trained model, so do no more training here


class lample:
    def __init__(self, dir, nclasses):
        self.dir = dir # where we store the model
        self.nclasses = nclasses

    def train(self, text, doc_start, labels, detext, dedoc_start, delabels):

        best_dev = -np.inf

        for seed in np.random.randint(1, 100000, size=1): # previously tried 10 seeds

            np.random.seed(seed)
            random.seed(seed)

            labeller = LSTMWrapper(self.dir, embpath)

            train_sentences, IOB_map, IOB_label = data_to_lstm_format(text, doc_start, labels.flatten(), self.nclasses)
            dev_sentences, _, _ = data_to_lstm_format(detext, dedoc_start, delabels.flatten(), self.nclasses)

            all_sentences = np.concatenate((train_sentences, dev_sentences), axis=0)

            _, _, dev_score = labeller.train_LSTM(all_sentences, train_sentences, dev_sentences, delabels,
                                                  IOB_map, IOB_label, self.nclasses, n_epochs=5, freq_eval=1,
                                                  max_niter_no_imprv=5, crf_probs=False) # previously let the epochs run to 100

            print('Dev score = %f' % dev_score)
            if dev_score > best_dev:
                best_model = labeller
                best_dev = dev_score
                print('--new best dev score.')

        self.labeller = best_model
        best_model.model.save()


    def predict(self, text, doc_start):
        Nte = len(text)
        test_sentences, _, _ = data_to_lstm_format(text, doc_start, np.ones(Nte), self.nclasses)
        preds_s, _ = self.labeller.predict_LSTM(test_sentences)
        return preds_s
