'''
Abstract class for the annotator models.
'''
import numpy as np
from scipy.special import gammaln

class Annotator():

    def update_post_alpha(self, E_t, C, doc_start, nscores):
        pass


    def update_post_alpha_data(self, model_idx, E_t, C, doc_start, nscores):
        pass


    def read_lnPi(self, l, C, Cprev, Krange, nscores, blanks):
        return self._read_lnPi(self.lnPi, l, C, Cprev, Krange, nscores, blanks)


    def read_lnPi_data(self, l, C, Cprev, nscores, model_idx):
        return self._read_lnPi(self.lnPi_data[model_idx], l, C, Cprev, 0, nscores, None)


    def read_lnEPi(self, l, C, Cprev, Krange, nscores, blanks):
        lnEPi = self.lnEPi()
        return self._read_lnPi(lnEPi, l, C, Cprev, Krange, nscores, blanks)


    def read_lnEPi_data(self, l, C, Cprev, nscores, model_idx):
        lnEPi_data = self.lnEPi_data()
        return self._read_lnPi(lnEPi_data[model_idx], l, C, Cprev, 0, nscores, None)


    def _read_lnPi(self, lnPi, l, C, Cprev, workeridxs, nscores, blanks):
        pass


    def q_pi(self):
        self.lnPi = self._calc_q_pi(self.alpha)


    def q_pi_data(self, model_idx):
        self.lnPi_data[model_idx] = self._calc_q_pi(self.alpha_data[model_idx])

    def _calc_q_pi(self, alpha):
        pass


    def lnEPi(self):
        EPi = self._calc_EPi(self.alpha)

        lnEPi = np.zeros_like(EPi)
        lnEPi[EPi != 0] = np.log(EPi[EPi != 0])
        lnEPi[EPi == 0] = -np.inf

        return lnEPi


    def lnEPi_data(self):
        if len(self.alpha0_data) == 0:
            return 0

        lnEPi_data = []

        for midx, _ in enumerate(self.alpha_data):
            EPi_m = self._calc_EPi(self.alpha_data[midx])
            lnEPi_m = np.zeros_like(EPi_m)
            lnEPi_m[EPi_m != 0] = np.log(EPi_m[EPi_m != 0])
            lnEPi_m[EPi_m == 0] = -np.inf
            lnEPi_data.append(lnEPi_m)

        return lnEPi_data


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


    def informativeness(self):

        ptj = np.zeros(self.L)
        for j in range(self.L):
            ptj[j] = np.sum(self.beta0[:, j]) + np.sum(self.Et == j)

        entropy_prior = -np.sum(ptj * np.log(ptj))

        ptj_c = np.zeros((self.L, self.L, self.K))
        for j in range(self.L):
            if self.alpha.ndim == 4:
                ptj_c[j] = np.sum(self.alpha[j, :, :, :], axis=1) / np.sum(self.alpha[j, :, :, :], axis=(0,1))[None, :] * ptj[j]
            elif self.alpha.ndim == 3:
                ptj_c[j] = self.alpha[j, :, :] / np.sum(self.alpha[j, :, :], axis=0)[None, :] * ptj[j]
            else:
                print('Warning: informativeness not defined for this annotator model.')

        ptj_giv_c = ptj_c / np.sum(ptj_c, axis=0)[None, :, :]

        entropy_post = -np.sum(ptj_c * np.log(ptj_giv_c), axis=(0,1))

        return entropy_prior - entropy_post


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