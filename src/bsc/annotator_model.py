'''
Abstract class for the annotator models.
'''
import numpy as np

class Annotator():

    def read_lnPi(self, l, C, Cprev, Krange, nscores, blanks=None):
        return self._read_lnPi(self.lnPi, l, C, Cprev, Krange, nscores, blanks)


    def read_lnPi_data(self, l, C, Cprev, Krange, nscores, model_idx, blanks=None):
        return self._read_lnPi(self.lnPi_data[model_idx], l, C, Cprev, Krange, nscores, blanks)


    def q_pi(self):
        self.lnPi = self._calc_q_pi(self.alpha)


    def q_pi_data(self, model_idx):
        self.lnPi_data[model_idx] = self._calc_q_pi(self.alpha_data[model_idx])


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