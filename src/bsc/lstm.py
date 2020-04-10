import datetime
import os

import numpy as np


class LSTM:

    train_type = 'MLE'#'Bayes'

    def __init__(self, model_dir, reload_model=False, embeddings_file=None):
        self.model_dir = model_dir
        self.reload_model = reload_model
        self.emb_file = embeddings_file
        self.completed_epochs = 0

    def init(self, N, text, doc_start, nclasses, max_vb_iters, crf_probs, dev_sentences, A, max_epochs=10):

        self.max_epochs = max_epochs # sets the total number of training epochs allowed. After this, it will just let the BSC
        #  model converge.
        self.n_epochs_per_vb_iter = 1

        self.crf_probs = crf_probs
        self.max_vb_iters = max_vb_iters

        self.N = N

        labels = np.zeros(N) # blank at this point. The labels get changed in each VB iteration

        from taggers.lample_lstm_tagger.lstm_wrapper import data_to_lstm_format, LSTMWrapper

        self.sentences, self.IOB_map, self.IOB_label = data_to_lstm_format(text, doc_start, labels, nclasses)

        timestamp = datetime.datetime.now().strftime('started-%Y-%m-%d-%H-%M-%S')
        self.model_dir = os.path.join(self.model_dir, './models_bac_%s' % timestamp)

        self.LSTMWrapper = LSTMWrapper(self.model_dir, embeddings=self.emb_file, reload_from_disk=self.reload_model)

        self.Ndocs = self.sentences.shape[0]

        self.train_data_objs = None

        self.nclasses = nclasses

        self.dev_sentences = dev_sentences
        if dev_sentences is not None:
            self.all_sentences = np.concatenate((self.sentences, self.dev_sentences))
            dev_gold = []
            for sen in self.dev_sentences:
                for tok in sen:
                    dev_gold.append( self.IOB_map[tok[1]] )
            self.dev_labels = dev_gold
        else:
            self.all_sentences = self.sentences

        self.tdev = np.zeros((len(self.dev_labels), self.nclasses))
        self.tdev[np.arange(self.tdev.shape[0]), self.dev_labels] = 1
        self.doc_start_dev = np.zeros((self.tdev.shape[0], 1))
        pointer = 0
        for sen in self.dev_sentences:
            self.doc_start_dev[pointer] = 1
            pointer += len(sen)

        self.A = A


    def fit_predict(self, labels):

        if self.train_type == 'Bayes':
            labels = np.argmax(labels, axis=1) # uses the mode

        print('Training LSTM on labels: ' + str(np.unique(labels)))
        print(np.sum(labels == 0))
        print(np.sum(labels == 1))
        print(np.sum(labels == 2))

        l = 0
        labels_by_sen = []
        for s, sen in enumerate(self.sentences):
            sen_labels = []
            labels_by_sen.append(sen_labels)
            for t, tok in enumerate(sen):
                self.sentences[s][t][1] = self.IOB_label[labels[l]]
                sen_labels.append(self.IOB_label[labels[l]])
                l += 1

        model_updated = False

        max_niter_no_imprv = 2
        freq_eval = 5

        if self.completed_epochs == 0:
            # the first update can use more epochs if we have allowed them
            n_epochs = self.max_epochs - ((self.max_vb_iters-1) * self.n_epochs_per_vb_iter) # the first iteration needs a bit more to move off the random initialisation
            if n_epochs < self.n_epochs_per_vb_iter:
                n_epochs = self.n_epochs_per_vb_iter

            # don't need to use an dev set here for early stopping as this may break EM
            self.lstm, self.f_eval, _ = self.LSTMWrapper.train_LSTM(self.all_sentences, self.sentences, self.dev_sentences,
                                                                 self.dev_labels, self.IOB_map, self.IOB_label,
                                                                 self.nclasses, n_epochs, freq_eval=freq_eval,
                                                                 crf_probs=self.crf_probs,
                                                                 max_niter_no_imprv=max_niter_no_imprv)

            self.completed_epochs = n_epochs
            model_updated = True

            self.niter_no_imprv = 0

        elif self.completed_epochs <= self.max_epochs:
            n_epochs = self.n_epochs_per_vb_iter  # for each bac iteration after the first

            best_dev = self.LSTMWrapper.best_dev
            last_score = self.LSTMWrapper.last_score

            for epoch in range(n_epochs):
                self.niter_no_imprv, best_dev, last_score = self.LSTMWrapper.run_epoch(0, self.niter_no_imprv,
                                                    best_dev, last_score,
                                                    compute_dev=(n_epochs>1) and ((epoch + self.completed_epochs + 1) % freq_eval)==0)

                model_updated = True

            # if self.LSTMWrapper.best_model_saved: # check that we are actually saving models before loading the best one so far
            #     self.LSTMWrapper.model.reload()

            self.completed_epochs += n_epochs

        # now make predictions for all sentences
        if model_updated:
            agg, probs = self.LSTMWrapper.predict_LSTM(self.sentences)
            self.probs = probs
            print('LSTM assigned class labels %s' % str(np.unique(agg)) )
        else:
            probs = self.probs

        # self.alpha0_data = self.A._post_alpha_data(self.tdev, self.dev_probs, self.alpha0_data_prior, self.alpha0_data,
        #                                            self.doc_start_dev, self.nclasses, -1)

        return probs

    def predict(self, doc_start, text):
        from taggers.lample_lstm_tagger.lstm_wrapper import data_to_lstm_format
        test_sentences, _, _ = data_to_lstm_format(text, doc_start, np.ones(N), self.nclasses)

        # now make predictions for all sentences
        agg, probs = self.LSTMWrapper.predict_LSTM(test_sentences)

        print('LSTM assigned class labels %s' % str(np.unique(agg)))

        return probs

    def log_likelihood(self, C_data, E_t):
        '''
        '''
        lnp_Cdata = C_data * np.log(E_t)
        lnp_Cdata[E_t == 0] = 0
        return np.sum(lnp_Cdata)