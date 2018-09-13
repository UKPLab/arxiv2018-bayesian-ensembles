import datetime

import numpy as np


class LSTM:

    train_type = 'Bayes'

    def init(self, alpha0_data, N, text, doc_start, nclasses, dev_data, max_vb_iters, crf_probs):

        self.crf_probs = crf_probs

        self.n_epochs_per_vb_iter = 1#3 # this may be too much?
        self.max_vb_iters = max_vb_iters

        self.probs = None

        self.N = N

        labels = np.zeros(N) # blank at this point. The labels get changed in each VB iteration

        from lample_lstm_tagger.lstm_wrapper import data_to_lstm_format, LSTMWrapper

        self.sentences, self.IOB_map, self.IOB_label = data_to_lstm_format(N, text, doc_start, labels, nclasses)

        timestamp = datetime.datetime.now().strftime('started-%Y-%m-%d-%H-%M-%S')
        self.LSTMWrapper = LSTMWrapper('./models_bac_%s' % timestamp)

        self.Ndocs = self.sentences.shape[0]

        self.train_data_objs = None

        self.nclasses = nclasses

        self.dev_sentences = dev_data
        if dev_data is not None:
            self.all_sentences = np.concatenate((self.sentences, self.dev_sentences))
            dev_gold = []
            for sen in self.dev_sentences:
                for tok in sen:
                    dev_gold.append( self.IOB_map[tok[1]] )
            self.dev_labels = dev_gold
        else:
            self.all_sentences = self.sentences

        alpha_data = np.copy(alpha0_data)
        self.alpha0_data = np.copy(alpha0_data)

        return alpha_data

    def fit_predict(self, labels, compute_dev_score=False):

        if self.train_type == 'Bayes':
            labels = np.argmax(labels, axis=1) # uses the mode

            # try sampling the posterior distribution. With enough iterations, this should work, but didn't in our tests with 20 iterations.
            #rvsunif = np.random.rand(labels.shape[0], 1)
            #labels = (rvsunif < np.cumsum(labels, axis=1)).argmax(1)

        l = 0
        labels_by_sen = []
        for s, sen in enumerate(self.sentences):
            sen_labels = []
            labels_by_sen.append(sen_labels)
            for t, tok in enumerate(sen):
                self.sentences[s][t][1] = self.IOB_label[labels[l]]
                sen_labels.append(self.IOB_label[labels[l]])
                l += 1

        # select a random subset of data to use for validation if the dataset is large
        if self.dev_sentences is None:

            devidxs = np.random.randint(0, self.Ndocs, int(np.round(self.Ndocs * 0.2)) )

            trainidxs = np.ones(self.Ndocs, dtype=bool)
            trainidxs[devidxs] = 0
            train_sentences = self.sentences[trainidxs]

            if len(devidxs) == 0:
                dev_sentences = self.sentences
            else:
                dev_sentences = self.sentences[devidxs]

            dev_labels = np.array(labels_by_sen)[devidxs].flatten()

        else:
            dev_sentences = self.dev_sentences
            train_sentences = self.sentences

            dev_labels = self.dev_labels

        freq_eval = 5
        max_niter_no_imprv = 2

        if self.LSTMWrapper.model is None:
            #from lample_lstm_tagger.lstm_wrapper import MAX_NO_EPOCHS

            # the first update needs more epochs to reach a useful level
            # n_epochs = MAX_NO_EPOCHS - ((self.max_vb_iters - 1) * self.n_epochs_per_vb_iter)
            # if n_epochs < self.n_epochs_per_vb_iter:
            #     n_epochs = self.n_epochs_per_vb_iter
            n_epochs = 20

            self.lstm, self.f_eval = self.LSTMWrapper.train_LSTM(self.all_sentences, train_sentences, dev_sentences,
                                                                 dev_labels, self.IOB_map, self.IOB_label,
                                                                 self.nclasses, n_epochs, freq_eval=freq_eval,
                                                                 crf_probs=self.crf_probs,
                                                                 max_niter_no_imprv=max_niter_no_imprv)
        else:
            n_epochs = self.n_epochs_per_vb_iter  # for each bac iteration after the first

            best_dev = -np.inf
            last_score = best_dev
            niter_no_imprv = 0

            self.LSTMWrapper.model.best_model_saved = False

            # for epoch in range(n_epochs):
            #     niter_no_imprv, best_dev, last_score = self.LSTMWrapper.run_epoch(0, niter_no_imprv,
            #                         best_dev, last_score, compute_dev_score and (((epoch+1) % freq_eval) == 0) and (epoch < n_epochs))

                # if niter_no_imprv >= max_niter_no_imprv:
                #     print("- early stopping %i epochs without improvement" % niter_no_imprv)
                #
                #     # Commented out because we're not sure what happens if we load a model from earlier iterations.
                #     # reload if something better was saved already.
                #         self.LSTMWrapper.model.reload()
                #     if self.LSTMWrapper.model.best_model_saved:
                #
                #     break

        # now make predictions for all sentences
        if self.probs is None:
            agg, self.probs = self.LSTMWrapper.predict_LSTM(self.sentences)

            print('LSTM assigned class labels %s' % str(np.unique(agg)) )

        return self.probs

    def predict(self, doc_start, text):
        from lample_lstm_tagger.lstm_wrapper import data_to_lstm_format
        N = len(doc_start)
        test_sentences, _, _ = data_to_lstm_format(N, text, doc_start, np.ones(N), self.nclasses)

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