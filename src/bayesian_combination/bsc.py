'''
Wrapper class to implement independent Bayesian sequence combination using the Bayesian combination framework.

'''
from bayesian_combination.bayesian_combination import BC


class BSC(BC):
    def __init__(self, L=3, K=5, max_iter=20, eps=1e-4, inside_labels=[0], outside_label=1, beginning_labels=[2],
                 alpha0_diags=1.0, alpha0_factor=1.0, alpha0_B_factor=1.0, beta0_factor=1.0,
                 annotator_model='seq', taggers=None, tagging_scheme='IOB2',
                 D=1, emission_model='cat',model_dir=None, embeddings_file=None, verbose=False):

        super().__init__(L, K, max_iter, eps, inside_labels, outside_label, beginning_labels,
                         alpha0_diags, alpha0_factor, alpha0_B_factor, beta0_factor,
                         annotator_model, taggers, tagging_scheme, true_label_model='HMM', discrete_feature_likelihoods=True,
                         D=D, emission_model=emission_model, model_dir=model_dir, embeddings_file=embeddings_file,
                         verbose=verbose, use_lowerbound=True)