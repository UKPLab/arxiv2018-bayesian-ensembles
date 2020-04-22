import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import logsumexp
from scipy.special.basic import psi

from bayesian_combination.tagger_wrappers.tagger import Tagger


class IndependentFeatures(Tagger):

    def __init__(self, nclasses, features):
        self.train_type = 'Bayes'
        self.nclasses = nclasses
        self.feat_map = {}
        self.features = []

        self.N = len(features)

        for feat in features.flatten():
            if feat not in self.feat_map:
                self.feat_map[feat] = len(self.feat_map)

            self.features.append( self.feat_map[feat] )

        self.features = np.array(self.features).astype(int)

        # sparse matrix of one-hot encoding, nfeatures x N
        self.features_mat = coo_matrix((np.ones(len(features)), (self.features, np.arange(self.N)))).tocsr()

        self.beta0 = np.ones((len(self.feat_map), self.nclasses)) * 0.001


    def fit_predict(self, Et):

        # count the number of occurrences for each label value

        beta = self.beta0 +  self.features_mat.dot(Et)

        self.ElnRho = psi(beta) - psi(np.sum(beta, 0)[None, :])

        lnptext_given_t = self.ElnRho[self.features, :]

        # normalise, assuming equal prior here
        pt_given_text = np.exp(lnptext_given_t - logsumexp(lnptext_given_t, 1)[:, None])

        return pt_given_text


    def predict(self, doc_start, features):
        N = len(doc_start)

        test_features = np.zeros(N, dtype=int)
        valid_feats = np.zeros(N, dtype=bool)

        for i, feat in enumerate(features.flatten()):
            if feat in self.feat_map:
                valid_feats[i] = True
                test_features[i] = self.feat_map[feat]

        lnptext_given_t = self.ElnRho[test_features[valid_feats], :]

        # normalise, assuming equal prior here
        pt_given_text = np.exp(lnptext_given_t - logsumexp(lnptext_given_t, 1)[:, None])

        probs = np.zeros((N, self.nclasses))
        probs[valid_feats, :] = pt_given_text

        return probs


    def log_likelihood(self, C_data, E_t):

        lnptext_given_t = self.ElnRho[self.features, :]

        lnp_Cdata = E_t * lnptext_given_t
        lnp_Cdata[E_t == 0] = 0
        return np.sum(lnp_Cdata)