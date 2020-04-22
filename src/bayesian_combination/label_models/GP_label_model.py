from sklearn.feature_extraction.text import CountVectorizer

from bayesian_combination.label_models.label_model import LabelModel, IndependentLabelModel
import numpy as np

from taggers.gp_classifier_svi import GPClassifierSVI
from taggers.gp_classifier_vb import compute_median_lengthscales


class GPCLabelModel(IndependentLabelModel):
    def __init__(self, beta0, L, verbose=False):
        self.L = L
        self.verbose = verbose

        self.beta0 = beta0
        self.beta_shape = beta0.shape

        self.nclasses = L
        self.gpc = {}


    def set_beta0(self, beta0):
        self.beta0 = beta0
        self.gpc = {}


    def update_B(self, features, ln_features_likelihood, predict_only=False):

        if len(self.gpc) == 0: # new data for prediction
            self.N = len(features)

            if not np.issubdtype(features.dtype, np.number):
                # sparse matrix of one-hot encoding, nfeatures x N
                self.vectorizer = CountVectorizer()
                self.numerical_features = self.vectorizer.fit_transform(features.flatten()).toarray()
            else:
                self.numerical_features = features

            ls = compute_median_lengthscales(self.numerical_features)

            z0 = self.beta0 / np.sum(self.beta0)

            for l in range(self.L):

                # init a classifier for each class but one
                self.gpc[l] = GPClassifierSVI(self.numerical_features.shape[1], z0=z0[l], shape_s0=2, rate_s0=self.beta0[l] * 200,
                                              ls_initial=ls, verbose=False)
                self.gpc[l].n_converged = 3
                self.gpc[l].conv_threshold = 1e-2
                # self.gpc[l].max_iter_VB = 3

        if predict_only:
            self.N = len(features)

            if not np.issubdtype(features.dtype, np.number):
                # sparse matrix of one-hot encoding, nfeatures x N
                self.numerical_features = self.vectorizer.transform(features.flatten()).toarray()
            else:
                self.numerical_features = features

        post = np.zeros((self.numerical_features.shape[0], self.nclasses))

        for l in range(self.nclasses):
            if not predict_only: # don't train when in predict mode
                self.gpc[l].fit(self.numerical_features, self.Et[:, l], optimize=False, use_median_ls=False)
            post[:, l] = self.gpc[l].predict(self.numerical_features).flatten()

        post = post / np.sum(post, axis=1)[:, None]
        self.lnB = np.log(post) + ln_features_likelihood

        if np.any(np.isnan(self.lnB)):
            print('_update_B: nan value encountered!')


    def lowerbound_terms(self):
        lnpt = self.lnp_Ct * self.Et
        lnpt = np.sum(lnpt)

        lnqt = np.log(self.Et)
        lnqt[self.Et == 0] = 0
        lnqt *= self.Et
        lnqt = np.sum(lnqt)

        gp_elbo = 0
        for l in range(self.nclasses):
            gp_elbo += self.gpc[l].lowerbound()

        return lnpt - lnqt + gp_elbo