'''
Bayesian sequence combination, variational Bayes implementation.
'''
import numpy as np
from scipy.sparse.coo import coo_matrix
from scipy.special import logsumexp, psi, gammaln
from scipy.optimize.optimize import fmin
from joblib import Parallel, cpu_count, effective_n_jobs

from bayesian_combination.annotator_models.acc import AccuracyAnnotator
from bayesian_combination.annotator_models.cm import ConfusionMatrixAnnotator
from bayesian_combination.annotator_models.cv import ConfusionVectorAnnotator
from bayesian_combination.annotator_models.dynamic_cm import DynamicConfusionMatrixAnnotator
from bayesian_combination.label_models.GP_label_model import GPCLabelModel
from bayesian_combination.tagger_wrappers.gpc import GPC
from bayesian_combination.tagger_wrappers.indfeatures import IndependentFeatures
from bayesian_combination.label_models.label_model import IndependentLabelModel
from bayesian_combination.label_models.markov_label_model import MarkovLabelModel
from bayesian_combination.tagger_wrappers.lstm import LSTM
from bayesian_combination.annotator_models.spam import SpamAnnotator
from bayesian_combination.annotator_models.seq import SequentialAnnotator


n_jobs = int(effective_n_jobs() / 2)  # limit the number of parallel jobs. Within each job, numpy can spawn more threads,
# which can cause the total number of CPUs required to exceed the limit.


class BC(object):

    def __init__(self, L=3, K=5, max_iter=20, eps=1e-4, inside_labels=[0], outside_label=1, beginning_labels=[2],
                 alpha0_diags=1.0, alpha0_factor=1.0, alpha0_B_factor=1.0, beta0_factor=1.0, nu0=1,
                 annotator_model='ibcc', taggers=None, tagging_scheme='IOB2', true_label_model='HMM',
                 discrete_feature_likelihoods=True, model_dir=None, embeddings_file=None,
                 verbose=False, use_lowerbound=True, converge_workers_first=False):
        """
        Aggregation method for combining labels from multiple annotators using variational Bayes. Different
        configurations of this class can be used to implement models such as IBCC-VB, MACE, BCCWords, DynIBCC, heatmapBCC
        and BSC-seq. The annotators may be experts, crowds, or automated labelling methods. It can also train a new
        automated tagger directly from the annotations (no gold standard) using variational combined supervision (VCS).

        The method can be applied to either classification tasks (with independent data points) or sequence labelling.

        :param L: number of labels/classes. E.g., for binary classification, L=2, for BIO sequence labelling, L=3.
        :param K: number of annotators.
        :param max_iter: maximum number of VB iterations. If time is limited, it can be good to cap this.
        :param eps: convergence threshold.
        :param inside_labels: for sequence labelling with BIO/IOB tagging schemes only, this indicates which classes
        correspond to I-tags.
        :param outside_label: for sequence labelling with BIO/IOB tagging schemes only, this indicates which class
        corresponds to an O tag.
        :param beginning_labels: for sequence labelling with BIO/IOB taggin schemes only, this indicates which classes
        correspond to B- tags.
        :param alpha0_diags: hyperparameter for annotator reliability model. This controls our prior confidence in the
        annotators' accuracy, so increasing it relative to alpha0_factor puts more initial weight on all annotators.
        Increasing magnitude also strengthens overall confidence in the prior, so small values are recommended.
        :param alpha0_factor: hyperparameter for annotator reliability model. This controls our prior belief in the
        annotators' randomness, so increasing it relative to alpha0_diags smoothes the reliability model and avoids
        putting much weight on the annotators. Increasing magnitude also strengthens overall confidence in the prior,
        so small values are recommended.
        :param alpha0_B_factor: hyperparameters for the seq annotator model that increases prior confidence in the
        annotators' B- labels.
        :param beta0_factor: hyperparameters for the label model. This acts as smoothing, so increasing the value
        of beta0_factor increases the weight of the prior and causes a more random distribution over the labels.
        :param nu0: hyperparameter for the independent feature model. This acts as smoothing, decreasing the weight
        of learned values relative to the prior.
        :param annotator_model: model for the annotators' reliability, can be 'acc' (learns accuracy only),
        'spam' (learns a MACE-like model of spamming patterns), 'CV' (confusion vector as in HMM-crowd),
        'CM' (confusion matrix, as in Dawid and Skene 1979), 'seq' (sequential confusion matrix, as in BSC-seq),
        'dyn.initial' (dynamic confusion matrix, as in DynIBCC) or 'dyn.consistent' (a variant of DynIBCC that applies
        the priors to all data points, rather than just the first annotation in the sequence).
        :param taggers: a list [] of names of automated taggers to be trained using VCS. Can be 'LSTM' (BiLSTM-CRF model
        of Lample et al. 2016), 'GP' (Gaussian process classifier method) or 'IF' (independent features similar to
        Naive Bayes).
        :param tagging_scheme: For sequential data, may be 'none',
        'IOB' (B token is used to separate consecutive spans)
        or 'IOB2' (also known as BIO, all spans start with a B-).
        If tagging scheme is 'none', then there will be no use of the parameters inside_labels, outside_label,
        beginning_labels, or alph0_B_factor.
        :param true_label_model: Can be 'HMM' (sequential label model used in BSC-seq) or 'GP' (Gaussian process label
        model used in heatmapBCC) or None (independent labels, as in IBCC or Dawid and Skene).
        :param discrete_feature_likelihoods: if False, the Bayesian combination method will not compute independent
         feature likelihoods. Use this setting if there are no features, if you wish to use the taggers or label model
         to account for features, or to reproduce the IBCC, DynIBCC, or MACE methods. If set to True, the likelihood
         of the features of each data point are included when computing its class distribution. Use this to reproduce
         BCCWords or BSC-seq.
        :param model_dir: Location used by taggers to cache model data.
        :param embeddings_file: Location of embeddings used by taggers.
        :param verbose: turns on some additional progress messages.
        :param use_lowerbound: if True, it uses the evidence lower bound (ELBO or variational lower bound) to determine
        convergence. If this is set to False, convergence is determined from the target labels, which is quicker to compute.
        :param converge_workers_first: If True, taggers will not be trained and will be excluded until the model has
        converged. This is a typical way to initialise VCS. If False, the taggers will be trained from the first iteration.
        """

        self.verbose = verbose
        # tagging_scheme can be 'IOB2' (all annos start with B) or 'IOB' (only annos
        # that follow another start with B).
        self.L = L  # number of classes
        if tagging_scheme == 'none':
            # Sequence tagging models need to select the correct sets of probabilities at the start of the sequence,
            # i.e., where there is no previous label. We use the outside label to indicate the index of these prior
            # probabilities within the label model or annotator model. Setting the value to L means that sequential
            # label/annotator models will create an extra index for these priors, and use L to reference that index.
            # BIO/IOB tagging schemes will just use the standard outside 'O' class, rather then creating separate priors
            # for the start of the sequence.
            outside_label = L

            inside_labels = []
            beginning_labels = []

        self.K = K  # number of annotators

        # Settings for the VB algorithm:
        self.max_iter = max_iter  # maximum number of VB iterations
        self.eps = eps  # threshold for convergence
        self.iter = 0  # current VB iteration
        self.max_internal_iters = 20
        self.converge_workers_first = converge_workers_first

        self.use_lb = use_lowerbound
        if self.use_lb:
            self.lb = -np.inf

        if self.verbose:
            print('Parallel can run %i jobs simultaneously, with %i cores' % (effective_n_jobs(), cpu_count()))

        # Do we model word distributions as independent features?
        self.no_features = not discrete_feature_likelihoods
        self.nu0 = nu0 # prior hyperparameter for word features

        # True label model: choose whether to use the HMM transition model.
        if true_label_model == 'HMM':
            if tagging_scheme == 'none':
                # record initial state probabilities as an additional row in the transition matrix, so use L+1 here
                num_prev_classes = self.L+1
            else:
                num_prev_classes = self.L
            self.LM = MarkovLabelModel(np.ones((num_prev_classes, self.L)) * beta0_factor, self.L, verbose,
                                       tagging_scheme, beginning_labels, inside_labels,
                                       outside_label)
        elif true_label_model == 'GP':
            self.LM = GPCLabelModel(np.ones(self.L) * beta0_factor, self.L, verbose)
        else:
            self.LM = IndependentLabelModel(np.ones(self.L) * beta0_factor, self.L, verbose)

        self.Et = None  # current true label estimates
        self.Et_old = None  # previous true labels estimates

        # options related to automated tagging models
        self.max_data_updates_at_end = 0  # no data model to be updated after everything else has converged.
        self.tagger_names = taggers if taggers is not None else []
        N_taggers = len(self.tagger_names)
        self.model_dir = model_dir # directory for caching the model
        self.embeddings_file = embeddings_file # pretrained word embeddings

        # choose type of worker model
        annotator_model = annotator_model.lower()
        if annotator_model == 'acc':
            self.A = AccuracyAnnotator(alpha0_diags, alpha0_factor, L, N_taggers)

        elif annotator_model == 'mace' or annotator_model == 'spam':
            self.A = SpamAnnotator(alpha0_diags, alpha0_factor, L, N_taggers)

        elif annotator_model == 'ibcc' or annotator_model == 'cm':
            self.A = ConfusionMatrixAnnotator(alpha0_diags, alpha0_factor, L, N_taggers)

        elif annotator_model == 'seq':
            self.A = SequentialAnnotator(alpha0_diags, alpha0_factor, L, N_taggers, tagging_scheme,
                                         beginning_labels, inside_labels, outside_label, alpha0_B_factor)

        elif annotator_model == 'vec' or annotator_model == 'cv':
            self.A = ConfusionVectorAnnotator(alpha0_diags, alpha0_factor, L, N_taggers)

        elif 'dyn' == annotator_model.split('.')[0]:

            toks = annotator_model.split('.')
            if len(toks)>1:
                prior_type = toks[1]
                self.A = DynamicConfusionMatrixAnnotator(alpha0_diags, alpha0_factor, L, N_taggers, prior_type)
            else:
                self.A = DynamicConfusionMatrixAnnotator(alpha0_diags, alpha0_factor, L, N_taggers)

        else:
            print('Bayesian combination: Did not recognise the annotator model %s' % annotator_model)

    def fit_predict(self, C, doc_start=None, features=None, dev_sentences=[], gold_labels=None):
        """
        Fits the Bayesian combination model to a set of annotations from multiple annotators (e.g.,
         crowdsourced annotations). It then predicts the gold-standard labels for these data points.
        The method works by running variational Bayesian approximate inference.

        :param C: A matrix of annotations from the crowd/ensemble of annotators. Rows correspond to data points, and
        columsn to annotators. For sequence labelling of text, the data points are individual tokens.
        Use -1 to indicate where the annotations were not provided by a particular annotator.
        :param doc_start: For sequence labelling, this is a binary vector with ones indicating the start of each
        sequence of labels (or document). For classification with independent data points, this should be None.
        :param features: (optional) a list containing a feature or feature vector for each data point.
        If the model was initialised with discrete_feature_likelihoods=True, there must be one discrete feature
        for each data point. For example, if the rows of C correspond to tokens in a series of documents,
        features contains that concatenated list of
        tokens for those documents. The GP label model is able to make use of distributed features (real-valued vectors),
        which can be passed here instead.
        :param dev_sentences: if using the LSTM tagger, dev set sentences can be passed in here to enable early stopping.
        See lample_lstm_tagger.lstm_wrapper.data_to_lstm_format() for a function to convert the data to the required format.
        :param gold_labels: an optional sequence of gold labels, with -1 for any missing values that should be inferred.
        Passing in gold labels helps to infer the annotator model and label model, so can improve performance over
        using the method in a purely unsupervised manner.
        :return: posterior distribution over the aggregated labels (array of size LxN), most likely aggregated labels
        (array of size N), probability of the most likely labels (array of size N).
        """
        if self.verbose:
            print('BSC: run() called with annotation matrix with shape = %s' % str(C.shape))

        # Initialise for this dataset------------------------------------------------------------------------

        # This function is called multiple times by optimize(), which modifies A and B but nothing else
        if self.iter == 0:
            self._init_dataset(gold_labels, C, features, doc_start)
        else:
            self.iter = 0 # got to reset this otherwise it won't learn again.

        # initialise the hyperparameters to correct sizes
        self.A.expand_alpha0(self.C, self.K, self.doc_start, self.L)
        self.A.init_lnPi(self.N)

        # Reset the taggers each time this is called
        self._init_taggers(features, dev_sentences)
        self.workers_converged = False

        # Variational Bayes inference loop -----------------------------------------------------------------------------
        with Parallel(n_jobs=n_jobs, backend='threading') as parallel:

            while not self._converged() or not self.workers_converged:
                if self.verbose:
                    print("BC iteration %i in progress" % self.iter)

                # Update true labels
                self.Et_old = self.Et
                if self.iter > 0:
                    self.Et = self.LM.update_t(parallel, self.C_data)
                else:
                    self.Et = self.LM.init_t(self.C, self.doc_start, self.A, self.blanks, self.gold)

                if self.verbose:
                    print("BC iteration %i: computed label probabilities" % self.iter)

                # Update label model and word feature distributions
                ln_feature_likelihood = self._update_features()
                self.LM.update_B(features, ln_feature_likelihood)
                if self.verbose:
                    print("BC iteration %i: updated label model" % self.iter)

                # Update automated taggers
                for midx, tagger in enumerate(self.taggers):
                    if not self.converge_workers_first or self.workers_converged:
                        # Update the data model by retraining the integrated tagger and obtaining its predictions
                        if tagger.train_type == 'Bayes':
                            self.C_data[midx] = tagger.fit_predict(self.Et)
                        elif tagger.train_type == 'MLE':
                            labels = self.LM.most_likely_labels(self._feature_ll(self.features))[1]
                            self.C_data[midx] = tagger.fit_predict(labels)

                        if self.verbose:
                            print('Tagger %s predicted the labels: %s' % (type(tagger), str(np.unique(np.argmax(
                                self.C_data[midx], axis=1)))))

                    if not self.converge_workers_first or self.workers_converged:
                        self.A.update_alpha_taggers(midx, self.Et, self.C_data[midx], self.doc_start, self.L)
                        self.A.q_pi_taggers(midx)
                        if self.verbose:
                            print("BC iteration %i: updated tagger of type %s" % (self.iter, str(type(tagger))) )

                if not self.converge_workers_first or self.workers_converged:
                    self.tagger_updates += 1  # count how many times we updated the data models

                # Update annotator models
                self.A.update_alpha(self.Et, self.C, self.doc_start, self.L)
                self.A.q_pi()
                if self.verbose:
                    print("BC iteration %i: updated worker models" % self.iter)

                # Increase iteration number
                self.iter += 1

            # VB has converged or quit -- now, compute the most likely sequence
            if self.verbose:
                print("BC iteration %i: computing most likely labels..." % self.iter)
            self.Et = self.LM.update_t(parallel, self.C_data)  # update t here so that we get the latest C_data and Et
            # values when computing most likely sequence and when returning Et
            plabels, labels = self.LM.most_likely_labels(self._feature_ll(self.features))

        if self.verbose:
            print("BC iteration %i: fitting/predicting complete." % self.iter)

        return self.Et, labels, plabels

    def predict(self, doc_start, features):
        '''
        If the model was initialised to include taggers, it can be used to make predictions on data with no annotations.
        This method should be called only after fit_predict().

        :param doc_start: a binary vector indicating the start of each sequence for sequence labelling, or None for
        classification with independent data points.
        :param features: a series of features for each test data point, of the same length as doc_start. The type of
        features must correspond to the type used in fit_predict.
        :return: posterior over the labels (array of shape LxN), most likely labels (array of shape L).
        '''

        if doc_start is None:
            # treat all the points as independent
            doc_start = np.ones(self.N)
        doc_start = doc_start.astype(bool)

        self.LM.C_by_doc = None

        with Parallel(n_jobs=n_jobs, backend='threading') as parallel:

            doc_start = doc_start.astype(bool)
            self.C = np.zeros((len(doc_start), self.K), dtype=int) - 1  # all blank
            self.blanks = self.C == 0

            C_data = []
            for model in self.taggers:
                C_data.append(model.predict(doc_start, features))

            self.LM.init_t(self.C, doc_start, self.A, self.blanks)
            ln_feature_likelihood = self._predict_features(doc_start, features)
            self.LM.update_B(features, ln_feature_likelihood, predict_only=True)
            q_t = self.LM.update_t(parallel, C_data)

            labels = self.LM.most_likely_labels(self._feature_ll(self.features))[1]

        return q_t, labels

    def optimize(self, C, doc_start, features=None, dev_sentences=[],
                    gold_labels=None, C_data_initial=None, maxfun=50):
        '''
        Run with MLII optimisation over the lower bound on the log-marginal likelihood.
        Optimizes the confusion matrix prior to the same values for all previous labels,
        and the scaling of the transition matrix hyperparameters.
        '''
        self.opt_runs = 0

        def neg_marginal_likelihood(hyperparams, C, doc_start):
            # set hyperparameters

            n_alpha_elements = len(hyperparams) - 1

            self.A.alpha0 = np.exp(hyperparams[0:n_alpha_elements]).reshape(self.A.alpha_shape)
            self.LM.set_beta0(np.ones(self.LM.beta_shape) * np.exp(hyperparams[-1]))

            # run the method
            self.fit_predict(C, doc_start, features, dev_sentences, gold_labels, C_data_initial)

            # compute lower bound
            lb = self.lowerbound()

            print("Run %i. Lower bound: %.5f, alpha_0 = %s, nu0 scale = %.3f" % (self.opt_runs,
                                                 lb, str(np.exp(hyperparams[0:-1])), np.exp(hyperparams[-1])))

            self.opt_runs += 1
            return -lb

        initialguess = np.log(np.append(self.A.alpha0.flatten(), self.LM.beta0.flatten()[0]))
        ftol = 1.0  #1e-3
        opt_hyperparams, _, _, _, _ = fmin(neg_marginal_likelihood, initialguess, args=(C, doc_start), maxfun=maxfun,
                                                     full_output=True, ftol=ftol, xtol=1e100)

        print("Optimal hyper-parameters: alpha_0 = %s, nu0 scale = %s" % (np.array2string(np.exp(opt_hyperparams[:-1])),
                                                                          np.array2string(np.exp(opt_hyperparams[-1]))))

        return self.Et, self.LM.most_likely_labels(self._feature_ll(self.features))[1]

    def lowerbound(self):
        '''
        Compute the variational lower bound (ELBO) on the log marginal likelihood.
        '''
        lnq_Cdata = 0

        for midx, Cdata in enumerate(self.C_data):
            lnq_Cdata_m = Cdata * np.log(Cdata)
            lnq_Cdata_m[Cdata == 0] = 0
            lnq_Cdata += lnq_Cdata_m

        lnq_Cdata = np.sum(lnq_Cdata)

        lnpPi, lnqPi = self.A.lowerbound_terms()

        lnpCtB, lnqtB = self.LM.lowerbound_terms()

        if self.no_features:
            lnpRho = 0
            lnqRho = 0
        else:
            lnpwords = np.sum(self.ElnRho[self.features, :] * self.Et)
            nu0 = np.zeros((1, self.L)) + self.nu0
            lnpRho = np.sum(gammaln(np.sum(nu0)) - np.sum(gammaln(nu0)) + sum((nu0 - 1) * self.ElnRho, 0)) + lnpwords
            lnqRho = np.sum(gammaln(np.sum(self.nu,0)) - np.sum(gammaln(self.nu),0) + sum((self.nu - 1) * self.ElnRho, 0))

        lb = lnpCtB - lnq_Cdata + lnpPi - lnqPi - lnqtB + lnpRho - lnqRho
        if self.verbose:
            print('Computing LB=%.4f: label model and labels=%.4f, annotator model=%.4f, features=%.4f' % (lb, lnpCtB-lnq_Cdata-lnqtB,
                                                                lnpPi-lnqPi, lnpRho-lnqRho))

        return lb

    def _init_features(self, features):
        '''
        This built-in feature model treats features as discrete, not distributed representations (embeddings). To handle
        distances between feature vectors, it is better to set no_words=True and use a GP label model.
        :param features:
        :return:
        '''

        self.feat_map = {} # map features tokens to indices
        self.features = []  # list of mapped index values for each token

        if self.no_features:
            return

        N = len(features)

        for feat in features.flatten():
            if feat not in self.feat_map:
                self.feat_map[feat] = len(self.feat_map)

            self.features.append( self.feat_map[feat] )

        self.features = np.array(self.features).astype(int)

        # sparse matrix of one-hot encoding, nfeatures x N, where N is number of tokens in the dataset
        self.features_mat = coo_matrix((np.ones(len(features)), (self.features, np.arange(N)))).tocsr()

    def _update_features(self):
        if self.no_features:
            return np.zeros((self.N, self.L))

        self.nu = self.nu0 + self.features_mat.dot(self.Et) # nfeatures x nclasses

        # update the expected log word likelihoods
        self.ElnRho = psi(self.nu) - psi(np.sum(self.nu, 0)[None, :])

        # get the expected log word likelihoods of each token
        lnptext_given_t = self.ElnRho[self.features, :]
        lnptext_given_t -= logsumexp(lnptext_given_t, axis=1)[:, None]

        return lnptext_given_t # N x nclasses where N is number of tokens/data points

    def _predict_features(self, doc_start, features):
        if self.no_features:
            lnptext_given_t = np.zeros((len(doc_start), self.L))
            self.features = None
        else:
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

    def _feature_ll(self, features):
        if self.no_features:
            features_ll = None
        else:
            ERho = self.nu / np.sum(self.nu, 0)[None, :]  # nfeatures x nclasses
            lnERho = np.zeros_like(ERho)
            lnERho[ERho != 0] = np.log(ERho[ERho != 0])
            lnERho[ERho == 0] = -np.inf
            features_ll = lnERho[features, :]

        return features_ll

    def _init_dataset(self, gold_labels, C, features, doc_start):
        # Save the data, init word model, and init true label array.
        # If there are any gold labels, supply a vector or list of values with -1 for the missing ones
        if gold_labels is not None:
            self.gold = np.array(gold_labels).flatten().astype(int)
        else:
            self.gold = None

        # initialise word model and worker models
        self._init_features(features)

        # initialise variables
        self.Et_old = np.zeros((C.shape[0], self.L))
        self.Et = np.ones((C.shape[0], self.L))

        # oldlb = -np.inf

        self.N = C.shape[0]

        if doc_start is None:
            # treat all the points as independent
            doc_start = np.ones(self.N)
        doc_start = doc_start.astype(bool)
        if doc_start.ndim == 1:
            doc_start = doc_start[:, None]
        self.doc_start = doc_start
        self.C = C.astype(int)
        self.blanks = C == -1

    def _init_taggers(self, features, dev_sentences):
        # initialise the integrated taggers
        self.C_data = []
        self.taggers = []
        for midx, tagger_name in enumerate(self.tagger_names):
            tagger_names = tagger_name.split('_')
            optionstr = tagger_names[1] if len(tagger_names)>1 else ''
            modelstr = tagger_names[0]

            if modelstr == 'LSTM':
                self.max_data_updates_at_end = 3 #20  # allow the other parameters to converge first,
                # then update LSTM with small number of iterations to avoid overfitting
                self.taggers.append(LSTM(self.L, features, self.doc_start, dev_sentences,
                    self.max_internal_iters, self.max_data_updates_at_end, True if 'crf' in optionstr else False,
                    self.model_dir, self.embeddings_file))

            elif modelstr == 'IF':
                self.taggers.append(IndependentFeatures(self.L, features))

            elif modelstr == 'GP':
                self.taggers.append(GPC(self.L, features))

            self.A.q_pi_taggers(midx)
            self.C_data.append(np.zeros((self.C.shape[0], self.L)))

        self.tagger_updates = 0

    def _convergence_diff(self):
        if self.use_lb:

            oldlb = self.lb
            self.lb = self.lowerbound()

            return self.lb - oldlb
        else:
            return np.max(np.abs(self.Et_old - self.Et))

    def _converged(self):
        '''
        Calculates whether the bayesian_combination has _converged or the maximum number of iterations is reached.
        The bayesian_combination has _converged when the maximum difference of an entry of q_t between two iterations is
        smaller than the given epsilon.
        '''
        if self.iter <= 1:
            return False # don't bother to check because some of the variables have not even been initialised yet!

        if self.workers_converged:
            # do we have to wait for data models to converge now?
            converged = (not self.converge_workers_first) or (self.tagger_updates >= self.max_data_updates_at_end)
        else:
            # have we run out of iterations for learning worker models?
            converged = self.iter >= self.max_iter

        # We have not reached max number of iterations yet. Check to see if updates have already converged.
        # Don't perform this check if the taggers has been updated once: it will not yet have resulted in changes to
        # self.Et, so diff will be ~0 but we need to do at least one more iteration.
        # If self.tagger_updates == 0 then data model is either not used or not yet updated, in which case, check
        # for convergence of worker models now.
        if not converged and self.tagger_updates != 1:

            diff = self._convergence_diff()

            if self.verbose:
                print("BSC: max. difference at iteration %i: %.5f" % (self.iter, diff))

            converged = diff < self.eps

        # The worker models have converged now, but we may still have data models to update
        if converged and not self.workers_converged:
            self.workers_converged = True
            if self.converge_workers_first:
                converged = False

        return converged


    def informativeness(self):

        lnB = self.LM.lnB
        while lnB.ndim > 1:
            lnB = np.sum(lnB, axis=0)
        ptj = lnB / np.sum(lnB)

        entropy_prior = -np.sum(ptj * np.log(ptj))

        ptj_c = np.zeros((self.L, self.L, self.K))
        for j in range(self.L):
            if self.A.alpha.ndim == 4:
                ptj_c[j] = np.sum(self.A.alpha[j, :, :, :], axis=1) / np.sum(self.A.alpha[j, :, :, :], axis=(0,1))[None, :] * ptj[j]
            elif self.A.alpha.ndim == 3:
                ptj_c[j] = self.A.alpha[j, :, :] / np.sum(self.A.alpha[j, :, :], axis=0)[None, :] * ptj[j]
            else:
                print('Warning: informativeness not defined for this annotator model.')

        ptj_giv_c = ptj_c / np.sum(ptj_c, axis=0)[None, :, :]

        entropy_post = -np.sum(ptj_c * np.log(ptj_giv_c), axis=(0,1))

        return entropy_prior - entropy_post
