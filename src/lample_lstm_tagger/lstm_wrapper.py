import os
import time
from collections import OrderedDict
import itertools

from sklearn.metrics import log_loss

from lample_lstm_tagger.utils import create_input
from lample_lstm_tagger.utils import models_path, evaluate, eval_script, eval_temp
from lample_lstm_tagger.loader import word_mapping, char_mapping, tag_mapping
from lample_lstm_tagger.loader import update_tag_scheme, prepare_dataset
from lample_lstm_tagger.loader import augment_with_pretrained
from lample_lstm_tagger.model import Model

import numpy as np


def data_to_lstm_format(nannotations, text, doc_start, labels, nclasses=0):
    sentences_list = []

    labels = labels.flatten()

    IOB_label = {1: 'O', 0: 'I-0', -1: 0}
    IOB_map = {'O': 1, 'I-0': 0}

    label_values = np.unique(labels) if nclasses==0 else np.arange(nclasses, dtype=int)

    for val in label_values:
        if val <= 1:
            continue

        IOB_label[val] = 'I-' if np.mod(val, 2) == 1 else 'B-'
        IOB_label[val] = IOB_label[val] + '-%i' % ((val - 3) / 2)

        IOB_map[IOB_label[val]] = val

    for i in range(nannotations):

        label = labels[i]

        if doc_start[i]:
            sentence_list = []
            sentence_list.append([text[i][0], IOB_label[label]])
            sentences_list.append(sentence_list)
        else:
            sentence_list.append([text[i][0], IOB_label[label]])

    sentences_list = np.array(sentences_list)

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

    return train_sentences, dev_sentences


def train_LSTM(all_sentences, train_sentences, dev_sentences, IOB_map, nclasses, n_epochs=100):

    # parameters
    parameters = OrderedDict()
    parameters['tag_scheme'] = 'iob'
    parameters['lower'] = 0
    parameters['zeros'] = 0
    parameters['char_dim'] = 25
    parameters['char_lstm_dim'] = 25
    parameters['char_bidirect'] = 1
    parameters['word_dim'] = 100
    parameters['word_lstm_dim'] = 100
    parameters['word_bidirect'] = 1
    parameters['pre_emb'] = "" # as per Nguyen 2017, which we compare against.
    # Would also be good to run with word embeddings included because:
    # - comparison against results with gold training data
    # -
    parameters['all_emb'] = 0
    parameters['cap_dim'] = 0
    parameters['crf'] = 1
    parameters['dropout'] = 0.5
    parameters['lr_method'] = "sgd-lr_.005"

    max_niter_no_imprv = 3
    #lr_decay = 0.9 # not currently used

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # Initialize model
    model = Model(parameters=parameters, models_path=models_path)# + '/' + model_name)
    print("Model location: %s" % model.model_path)

    # Data parameters
    lower = parameters['lower']
    #tag_scheme = parameters['tag_scheme']

    # Use selected tagging scheme (IOB / IOBES)
    # update_tag_scheme(train_sentences, tag_scheme)
    # update_tag_scheme(dev_sentences, tag_scheme)
    # update_tag_scheme(test_sentences, tag_scheme)

    # dico_words_train = word_mapping(train_sentences, lower)[0]
    # dico_words, word_to_id, id_to_word = augment_with_pretrained(
    #     dico_words_train.copy(),
    #     parameters['pre_emb'],
    #     list(itertools.chain.from_iterable(
    #         [[w[0] for w in s] for s in dev_sentences + test_sentences])
    #     ) if not parameters['all_emb'] else None
    # )

    # Following LSTM-CROWD, we don't load any pre-trained word embeddings.
    dico_words, word_to_id, id_to_word = word_mapping(all_sentences, lower)
    dico_words_train = dico_words

    # Create a dictionary and a mapping for words / POS tags / tags
    dico_chars, char_to_id, id_to_char = char_mapping(all_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(all_sentences)

    # Index data
    train_data = prepare_dataset(
        train_sentences, word_to_id, char_to_id, tag_to_id, lower
    )
    dev_data = prepare_dataset(
        dev_sentences, word_to_id, char_to_id, tag_to_id, lower
    )

    # Save the mappings to disk
    print('Saving the mappings to disk...')
    model.save_mappings(id_to_word, id_to_char, id_to_tag)

    # Build the model
    f_train, f_eval = model.build(**parameters)

    #
    # Train network
    #
    singletons = set([word_to_id[k] for k, v in list(dico_words_train.items()) if v == 1])

    freq_eval = 1000  # evaluate on dev every freq_eval steps
    best_dev = -np.inf
    count = 0

    niter_no_imprv = 0 # for early stopping

    train_data_objs = [train_data, singletons, parameters, f_train, f_eval, dev_sentences, dev_data, id_to_tag,
                                          dico_tags]

    for epoch in range(n_epochs):
        count, niter_no_imprv, best_dev = run_epoch(epoch, count, freq_eval, train_data, singletons, parameters, f_train, f_eval,
                                          niter_no_imprv, max_niter_no_imprv, dev_sentences, dev_data, id_to_tag,
                                          dico_tags, model, best_dev, IOB_map, nclasses)
        #lr *= lr_decay # can't find an easy way to taper the learning rate

    # reload the best model as saved in run_epoch
    model.reload()

    return model, f_eval, train_data_objs

def run_epoch(epoch, count, freq_eval, train_data, singletons, parameters, f_train, f_eval, niter_no_imprv,
              max_niter_no_imprv, dev_sentences, dev_data, id_to_tag, dico_tags, model, best_dev,
              IOB_map, num_classes):
    epoch_costs = []
    print("Starting epoch %i..." % epoch)

    for i, index in enumerate(np.random.permutation(len(train_data))):
        # loop over random permutations of the training data
        count += 1

        input = create_input(train_data[index], parameters, True, singletons)
        new_cost = f_train(*input)
        epoch_costs.append(new_cost)

        if i % 50 == 0 and i > 0 == 0:
            print("%i, cost average: %f" % (i, np.mean(epoch_costs[-50:])))
        if count % freq_eval == 0:
            # dev_score = evaluate(parameters, f_eval, dev_sentences,
            #                      dev_data, id_to_tag, dico_tags)

            # maximise log likelihood of dev data
            agg, probs = predict_LSTM(model, dev_sentences, f_eval, num_classes)
            dev_score = - log_loss(dev_data['tags'], probs)

            print("Score on dev: %.5f" % dev_score)
            if dev_score > best_dev:
                best_dev = dev_score
                print("New best score on dev.")
                print("Saving model to disk...")
                model.save()
            else:
                niter_no_imprv += 1
                if niter_no_imprv >= max_niter_no_imprv:
                    print("- early stopping %i epochs without improvement" % niter_no_imprv)
                    break

    print("Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs)))

    return count, niter_no_imprv, best_dev

def predict_LSTM(model, test_sentences, f_eval, num_classes, IOB_map=None):
    # Load existing model
    parameters = model.parameters
    lower = parameters['lower']

    # Load reverse mappings
    word_to_id, char_to_id, tag_to_id = [
        {v: k for k, v in list(x.items())}
        for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
    ]

    test_data = prepare_dataset(
        test_sentences, word_to_id, char_to_id, tag_to_id, lower
    )

    start = time.time()

    print('Tagging...')
    count = 0

    agg = []
    probs = np.empty((0, num_classes))  # np.zeros((agg.shape[0], self.num_classes))

    for test_data_sen in test_data:

        input = create_input(test_data_sen, parameters, False)

        # Decoding
        if parameters['crf']:
            y_preds = np.array(f_eval(*input))[1:-1]
        else:
            y_preds = f_eval(*input).argmax(axis=1)

        # TODO replace id_to_tag map by our own map from the experiment data loader stuff... then we can skip all this
        y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]

        probs_sen = np.zeros((len(y_preds), num_classes))
        for i, val in enumerate(y_preds):

            if IOB_map is not None:
                val = IOB_map[val]

            probs_sen[i, val] = 1
            agg.append(val)

        probs = np.concatenate((probs, probs_sen), axis=0)

        count += 1
        if count % 100 == 0:
            print(count)

    print('---- %i lines tagged in %.4fs ----' % (count, time.time() - start))

    agg = np.array(agg)

    return agg, probs