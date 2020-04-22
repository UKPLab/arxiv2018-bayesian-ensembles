'''
Wrapper classes to implement MACE (Hovy et al., 2013, https://www.aclweb.org/anthology/N13-1132/)
using the Bayesian combination framework, along with variants of that model.

Note that the priors we use are slightly different (for consistency with the other Bayesian combination methods here),
so results can differ from other MACE implementations.
'''
from bayesian_combination.bayesian_combination import BC


class MACE(BC):
    def __init__(self, L=3, K=5, max_iter=20, eps=1e-4,
                     alpha0_diags=1.0, alpha0_factor=1.0, beta0_factor=1.0,
                     verbose=False, use_lowerbound=True):

        super().__init__(L, K, max_iter, eps,
                         alpha0_diags=alpha0_diags, alpha0_factor=alpha0_factor, beta0_factor=beta0_factor,
                         annotator_model='spam', taggers=None, true_label_model='noHMM', discrete_feature_likelihoods=False,
                         verbose=verbose, use_lowerbound=use_lowerbound, converge_workers_first=True)