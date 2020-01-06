'''
Abstract class for the annotator models.
'''

class Annotator():

    def read_lnPi(self, l, C, Cprev, Krange, nscores, blanks=None):
        return self._read_lnPi(self.lnPi, l, C, Cprev, Krange, nscores, blanks)


    def read_lnPi_data(self, l, C, Cprev, Krange, nscores, model_idx, blanks=None):
        return self._read_lnPi(self.lnPi_data[model_idx], l, C, Cprev, Krange, nscores, blanks)


    def q_pi(self):
        self.lnPi = self._calc_q_pi(self.alpha)


    def q_pi_data(self, model_idx):
        self.lnPi_data[model_idx] = self._calc_q_pi(self.alpha_data[model_idx])


    def EPi(self):
        return self._calc_EPi(self.alpha)


    def EPi_data(self, model_idx):
        return self._calc_EPi(self.alpha_data[model_idx])