'''
Abstract class for the annotator models.
'''
import numpy as np
from scipy.special import gammaln

class Annotator():

    def init_lnPi(self, N):
        pass


    def expand_alpha0(self, C, K, doc_start, nscores):
        pass


    def update_alpha(self, E_t, C, doc_start, nscores):
        pass


    def update_alpha_taggers(self, model_idx, E_t, C, doc_start, nscores):
        pass


    def read_lnPi(self, l, C, Cprev, doc_id, Krange, nscores, blanks):
        pass


    def read_lnPi_taggers(self, l, C, Cprev, nscores, model_idx):
        pass


    def q_pi(self):
        self.lnPi = self._calc_q_pi(self.alpha)


    def q_pi_taggers(self, model_idx):
        self.lnPi_taggers[model_idx] = self._calc_q_pi(self.alpha_taggers[model_idx])


    def _calc_q_pi(self, alpha):
        pass


    def annotator_accuracy(self):
        if self.alpha.ndim == 3:
            annotator_acc = self.alpha[np.arange(self.L), np.arange(self.L), :] \
                        / np.sum(self.alpha, axis=1)
        elif self.alpha.ndim == 2:
            annotator_acc = self.alpha[1, :] / np.sum(self.alpha[:2, :], axis=0)
        elif self.alpha.ndim == 4:
            annotator_acc = np.sum(self.alpha, axis=2)[np.arange(self.L), np.arange(self.L), :] \
                        / np.sum(self.alpha, axis=(1,2))

        if self.beta.ndim == 2:
            beta = np.sum(self.beta, axis=0)
        else:
            beta = self.beta

        annotator_acc *= (beta / np.sum(beta))[:, None]
        annotator_acc = np.sum(annotator_acc, axis=0)

        return annotator_acc


def log_dirichlet_pdf(alpha, lnPi, sum_dim):
    x = (alpha - 1) * lnPi
    gammaln_alpha = gammaln(alpha)
    invalid_alphas = np.isinf(gammaln_alpha) | np.isinf(x) | np.isnan(x)
    gammaln_alpha[invalid_alphas] = 0  # these possibilities should be excluded
    x[invalid_alphas] = 0
    x = np.sum(x, axis=sum_dim)
    z = gammaln(np.sum(alpha, sum_dim)) - np.sum(gammaln_alpha, sum_dim)
    if not np.isscalar(z):
        z[np.isinf(z)] = 0
    return np.sum(x + z)
