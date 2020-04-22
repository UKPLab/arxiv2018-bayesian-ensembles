from scipy.sparse.coo import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

from bayesian_combination.tagger_wrappers.tagger import Tagger
from taggers.gp_classifier_svi import GPClassifierSVI
import numpy as np

from taggers.gp_classifier_vb import compute_median_lengthscales


class GPC(Tagger):

    def __init__(self, nclasses, features):
        self.train_type = 'Bayes'
        self.nclasses = nclasses

        self.N = len(features)

        if not np.issubdtype(features.dtype, np.number):
            # sparse matrix of one-hot encoding, nfeatures x N
            self.vectorizer = CountVectorizer()
            self.numerical_features = self.vectorizer.fit_transform(features.flatten()).toarray()
        else:
            self.numerical_features = features

        ls = compute_median_lengthscales(self.numerical_features)

        self.gpc = {}
        for l in range(nclasses):
            # init a classifier for each class but one
            self.gpc[l] = GPClassifierSVI(self.numerical_features.shape[1], z0=1/float(nclasses), shape_s0=2, rate_s0=200,
                                          ls_initial=ls, verbose=False)
            self.gpc[l].n_converged = 3
            self.gpc[l].conv_threshold = 1e-2
            # self.gpc[l].max_iter_VB = 3

    def fit_predict(self, Et):
        '''
        Fit the model to the current expected true labels.
        :param Et:
        :return:
        '''
        post = Et.copy()

        for l in range(self.nclasses):
            self.gpc[l].fit(self.numerical_features, Et[:, l], optimize=False, use_median_ls=False)
            post[:, l] = self.gpc[l].predict(self.numerical_features).flatten()

        post = post / np.sum(post, axis=1)[:, None]
        return post

    def predict(self, doc_start, features):
        '''
        Use the trained model to predict.
        :param doc_start:
        :param features:
        :return:
        '''
        N = len(doc_start)

        if not np.issubdtype(features.dtype, np.number):
            numerical_features = self.vectorizer.transform(features.flatten()).toarray()
        else:
            numerical_features = features

        post = np.zeros((N, self.nclasses))
        for l in range(self.nclasses):
            post[:, l] = self.gpc[l].predict(out_feats=numerical_features).flatten()

        post = post / np.sum(post, axis=1)[:, None]
        return post

    def log_likelihood(self, C_data, E_t):
        '''
        Compute the log-likelihood of the model's current predictions for the training set.
        :param C_data:
        :param E_t:
        :return:
        '''
        elbo = 0
        for l in range(self.nclasses):
            elbo += self.gpc[l].lowerbound()
        return