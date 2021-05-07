'''
Emission model for Categorical Distribution
(used for NLP case - text features)
'''
import numpy as np
from scipy.special import logsumexp, psi, gammaln
from scipy.sparse.coo import coo_matrix

from bayesian_combination.emission_models.emission_model import Emission


class Categorical(Emission):
    def __init__(self, L, D, nu0=1):
        self.L = L  # number of classes
        self.D = D  # feature dimension
        self.nu0 = nu0      # prior hyperparameter (scalar)

        self.features = None # stored features
        self.feat_map = None # prior hyperparameter for word features
        self.ElnRho = None
        return

    def init_features(self, features):
        self.feat_map = {}  # map features tokens to indices
        self.features = []  # list of mapped index values for each token

        N = len(features)

        for feat in features.flatten():
            if feat not in self.feat_map:
                self.feat_map[feat] = len(self.feat_map)

            self.features.append(self.feat_map[feat])

        self.features = np.array(self.features).astype(int)

        # sparse matrix of one-hot encoding, nfeatures x N, where N is number of tokens in the dataset
        self.features_mat = coo_matrix((np.ones(len(features)), (self.features, np.arange(N)))).tocsr()
        return

    def update_features(self, Et):
        self.nu = self.nu0 + self.features_mat.dot(Et)  # nfeatures x nclasses

        # update the expected log word likelihoods
        self.ElnRho = psi(self.nu) - psi(np.sum(self.nu, 0)[None, :])

        # get the expected log word likelihoods of each token
        lnptext_given_t = self.ElnRho[self.features, :]
        lnptext_given_t -= logsumexp(lnptext_given_t, axis=1)[:, None]

        return lnptext_given_t  # N x nclasses where N is number of tokens/data points


    def predict_features(self, features):
        # get the expected log word likelihoods of each token
        self.features = []  # list of mapped index values for each token
        available = []
        for feat in features.flatten():
            if feat not in self.feat_map:
                available.append(0)
                self.features.append(0)
            else:
                available.append(1)
                self.features.append(self.feat_map[feat])
        lnptext_given_t = self.ElnRho[self.features, :] + np.array(available)[:, None]
        lnptext_given_t -= logsumexp(lnptext_given_t, axis=1)[:, None]

        return lnptext_given_t


    def feature_ll(self, features):
        ERho = self.nu / np.sum(self.nu, 0)[None, :]  # nfeatures x nclasses
        lnERho = np.zeros_like(ERho)
        lnERho[ERho != 0] = np.log(ERho[ERho != 0])
        lnERho[ERho == 0] = -np.inf
        features_ll = lnERho[features, :]

        return features_ll

    def lowerbound(self, Et):
        lnpwords = np.sum(self.ElnRho[self.features, :] * self.Et)
        nu0 = np.zeros((1, self.L)) + self.nu0
        lnpRho = np.sum(gammaln(np.sum(nu0)) - np.sum(gammaln(nu0)) + sum((nu0 - 1) * self.ElnRho, 0)) + lnpwords
        lnqRho = np.sum(gammaln(np.sum(self.nu, 0)) - np.sum(gammaln(self.nu), 0) + sum((self.nu - 1) * self.ElnRho, 0))

        return lnpRho, lnqRho