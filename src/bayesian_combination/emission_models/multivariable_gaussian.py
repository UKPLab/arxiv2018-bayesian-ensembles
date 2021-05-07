'''
Emission model for Multivariable Gaussian
'''
import numpy as np
from scipy.special import logsumexp, psi, gammaln

from bayesian_combination.emission_models.emission_model import Emission


def student_t_lnpdf(x, mu, L, nu):
    D = x.shape[1]
    lnC1 = gammaln(D/2+nu/2) - gammaln(nu/2)
    lnC2 = 0.5*np.log(np.linalg.det(L)) - (D/2) * np.log(np.pi * nu)
    mah2 = np.diag((x-mu) @ L @ (x-mu).T)
    lnC3 = (-D/2 - nu/2) * np.log(1 + mah2/nu)
    return lnC1 + lnC2 + lnC3


class Multivariable_Gaussian(Emission):
    def __init__(self, L, D, nu0=1, beta0=1, m0=0, W0=1):
        self.L = L # number of classes
        self.D = D # feature dimension
        self.nu0 = nu0      # prior hyperparameter (scalar)
        self.beta0 = beta0  # prior hyperparameter (scalar)
        if hasattr(m0, '__len__'):
            self.m0 = np.array(m0)  # prior hyperparameter (vector (D,))
        else:
            self.m0 = np.ones(self.D) * m0
        if hasattr(W0, '__len__'):
            self.W0 = np.array(W0)  # prior hyperparameter (matrix (D x D)
        else:
            self.W0 = np.eye(self.D) * W0
        self.W0_inv = np.linalg.inv(self.W0)

        self.features = None # stored features
        self.ElnRho = None
        return

    def init_features(self, features):
        if self.D != features.shape[1]:
            raise ValueError(f'invalid shape of features, expected {self.D} columns but received {features.shape[1]} columns')
        self.features = np.array(features)
        return

    def update_features(self, Et):
        N = []
        x_ = []
        NS = []

        self.beta = []
        self.m = []
        self.nu =[]
        self.W = []

        self.La = []
        self.fd = []

        ElnRho = []
        for j in range(self.L):
            # obtain features' statistics
            N.append(np.sum(Et[:,j]))
            if N[j] == 0:
                x_.append(np.dot(Et[:,j], self.features) / np.inf)
            else:
                x_.append(np.dot(Et[:, j], self.features) / N[j])
            NS.append(self.features.T @ np.diag(Et[:,j]) @ self.features)

            # update posterior model parameters for Gauss-Wishart distribution
            self.beta.append(self.beta0 + N[j])
            self.nu.append(self.nu0 + N[j])
            self.m.append((self.beta0 * self.m0 + N[j] * x_[j]) / self.beta[j])
            self.W.append(np.linalg.inv( self.W0_inv + NS[j] + self.beta0 * N[j] / self.beta[j] * np.outer(x_[j] - self.m0, x_[j] - self.m0)))

            # update predictive posterior model parameters
            if (self.nu[j] + 1 - self.D) < 0:
                self.La.append(self.W[j] * 0.0) # avoid having Negative definitive La matrix, which causes NaN
            else:
                self.La.append((self.nu[j] + 1 - self.D) / (1 + self.beta[j]) * self.W[j])
            self.fd.append(self.nu[j] - 1 + self.D)

            # compute ElnRho
            ElnL = np.sum(psi((self.nu[j]+1+np.arange(1,self.D+1))/2)) + self.D * np.log(2.) + np.log( np.linalg.det(self.W[j]))
            Ecov = self.D / self.beta[j] + np.diag(self.nu[j] * (self.features - self.m[j][None,:]) @ self.W[j] @ (self.features - self.m[j][None,:]).T)
            ElnRho.append(ElnL - self.D/2.0 * np.log(2*np.pi) - 0.5 * Ecov)

        lnptext_given_t = np.array(ElnRho).T
        lnptext_given_t -= logsumexp(lnptext_given_t, axis=1)[:, None]
        self.ElnRho = lnptext_given_t

        return lnptext_given_t # N x nclasses where N is number of tokens/data points


    def predict_features(self, features):
        # get the expected log word likelihoods of each token
        self.features = np.array(features)
        # compute ERho
        ElnRho = []
        for j in range(self.L):
            ElnL = np.sum(psi((self.nu[j] + 1 + np.arange(1, self.D + 1)) / 2)) + self.D * np.log(2.) + np.log(np.linalg.det(self.W[j]))
            Ecov = self.D / self.beta[j] + self.nu[j] * (self.features - self.m[j][None, :]) @ self.W[j] @ (self.features - self.m[j][None, :]).T
            ElnRho.append(ElnL - self.D / 2.0 * np.log(2 * np.pi) - 0.5 * Ecov)
        lnptext_given_t = np.array(ElnRho).T
        lnptext_given_t -= logsumexp(lnptext_given_t, axis=1)[:, None]
        self.ElnRho = lnptext_given_t

        return lnptext_given_t  # N x nclasses where N is number of tokens/data points



    def feature_ll(self, features):
        features = np.array(features)
        features_ll = []
        for j in range(self.L):
            features_ll.append(student_t_lnpdf(features, self.m[j], self.La[j], self.fd[j]))
        features_ll = np.array(features_ll).T

        return features_ll


    def lowerbound(self, Et):
        lnpX = np.sum(self.ElnRho * Et)
        lnB0 = - self.nu0 / 2 * np.linalg.det(self.W0) \
               - self.nu0 * self.D / 2 * np.log(2) \
               - self.D * (self.D - 1) / 4 * np.log(np.pi) \
               - np.sum(gammaln((self.nu0 + 1 + np.arange(1, self.D + 1)) / 2))
        lnpMuLa = 0
        lnqMuLa = 0
        for j in range(self.L):
            lnLa_ = np.sum(psi((self.nu[j] + 1 + np.arange(1, self.D + 1)) / 2)) \
                    + self.D * np.log(2.) + np.log(np.linalg.det(self.W[j]))
            lnpMuLa += 0.5 * self.D * np.log(self.beta0 / 2 / np.pi) \
                       + 0.5 * (self.nu0 - self.D) * lnLa_ \
                       + 0.5 * self.beta0 * self.nu[j] * (self.m[j] - self.m0) @ self.W[j] @ (self.m[j] - self.m0) \
                       + lnB0 \
                       - 0.5 * self.nu[j] * np.trace(np.linalg.inv(self.W0) @ self.W[j])

            lnB = - self.nu[j] / 2 * np.linalg.det(self.W[j]) \
                  - self.nu[j] * self.D / 2 * np.log(2) \
                  - self.D * (self.D - 1) / 4 * np.log(np.pi) \
                  - np.sum(gammaln((self.nu[j] + 1 + np.arange(1, self.D + 1)) / 2))
            HqLa = - lnB \
                   - (self.nu[j] - self.D - 1) / 2 * lnLa_ \
                   + self.nu[j] * self.D / 2
            lnqMuLa += 0.5 * lnLa_ \
                       + self.D / 2 * np.log(self.beta[j] / 2 / np.pi) \
                       - self.D / 2 \
                       - HqLa

        lnpRho = lnpX + lnpMuLa
        lnqRho = lnqMuLa

        return lnpRho, lnqRho
