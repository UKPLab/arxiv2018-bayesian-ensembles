import datetime

import numpy as np


class LSTM:

    train_type = 'Bayes'

    def init(self, alpha0_data, N, text, doc_start, nclasses, max_vb_iters, crf_probs, dev_sentences, max_epochs=20):

        self.max_epochs = max_epochs # sets the total number of training epochs allowed. After this, it will just let the BSC
        #  model converge.
        self.n_epochs_per_vb_iter = 0

        self.crf_probs = crf_probs
        self.max_vb_iters = max_vb_iters

        self.N = N

        labels = np.zeros(N) # blank at this point. The labels get changed in each VB iteration

        from lample_lstm_tagger.lstm_wrapper import data_to_lstm_format, LSTMWrapper

        self.sentences, self.IOB_map, self.IOB_label = data_to_lstm_format(N, text, doc_start, labels, nclasses)

        timestamp = datetime.datetime.now().strftime('started-%Y-%m-%d-%H-%M-%S')
        self.LSTMWrapper = LSTMWrapper('./models_bac_%s' % timestamp)

        self.Ndocs = self.sentences.shape[0]

        self.train_data_objs = None

        self.nclasses = nclasses

        alpha_data = np.copy(alpha0_data)
        self.alpha0_data = np.copy(alpha0_data)

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

        return alpha_data

    def fit_predict(self, labels):

        if self.train_type == 'Bayes':
            labels = np.argmax(labels, axis=1) # uses the mode

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

        if self.LSTMWrapper.model is None:
            # the first update needs more epochs to reach a useful level -- use all allowed epochs
            n_epochs = self.max_epochs - ((self.max_vb_iters-1) * self.n_epochs_per_vb_iter) # the first iteration needs a bit more to move off the random initialisation
            if n_epochs < 1:
                n_epochs = 1

            # don't need to use an dev set here for early stopping as this may break EM
            self.lstm, self.f_eval = self.LSTMWrapper.train_LSTM(self.all_sentences, self.sentences, self.dev_sentences,
                                                                 self.dev_labels, self.IOB_map, self.IOB_label,
                                                                 self.nclasses, n_epochs, freq_eval=1,
                                                                 crf_probs=self.crf_probs,
                                                                 max_niter_no_imprv=n_epochs)

            self.completed_epochs = n_epochs
            model_updated = True

        elif self.completed_epochs <= self.max_epochs:
            n_epochs = self.n_epochs_per_vb_iter  # for each bac iteration after the first

            # best_dev = -np.inf
            # last_score = best_dev
            best_dev = self.LSTMWrapper.best_dev
            last_score = self.LSTMWrapper.last_score

            niter_no_imprv = 2

            for epoch in range(n_epochs):
                niter_no_imprv, best_dev, last_score = self.LSTMWrapper.run_epoch(0, niter_no_imprv,
                                    best_dev, last_score, compute_dev=True)

                model_updated = True

            if self.LSTMWrapper.best_model_saved: # check that we are actually saving models before loading the best one so far
                self.LSTMWrapper.model.reload()

            self.completed_epochs += n_epochs

        # now make predictions for all sentences
        if model_updated:
            agg, probs = self.LSTMWrapper.predict_LSTM(self.sentences)
            self.probs = probs
            print('LSTM assigned class labels %s' % str(np.unique(agg)) )
        else:
            probs = self.probs

        return probs

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