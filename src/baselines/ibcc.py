'''
@author: Edwin Simpson
'''
import sys, logging
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import psi, gammaln
from scipy.optimize import fmin, fmin_cobyla
from scipy.stats import gamma

class IBCC(object):
# Print extra debug info
    verbose = False
    keeprunning = True # set to false causes the combine_classifications method to exit without completing if another 
    # thread is checking whether IBCC is taking too long. Probably won't work well if the optimize_hyperparams is true.    
# Configuration for variational Bayes (VB) algorithm for approximate inference -------------------------------------
    # determine convergence by calculating lower bound? Quicker to set=False so we check convergence of target variables
    uselowerbound = False
    min_iterations = 1
    max_iterations = 500
    conv_threshold = 1e-5
    conv_check_freq = 2
    
# Data set attributes -----------------------------------------------------------------------------------------------
    discretedecisions = False  # If true, decisions are rounded to discrete integers. If false, you can submit undecided
    # responses as fractions between two classes. E.g. 2.3 means that 0.3 of the decision will go to class 3, and 0.7
    # will go to class 2. Only neighbouring classes can have uncertain decisions in this way.
    table_format_flag = False  
    nclasses = None
    nscores = None
    K = None
    N = 0 #number of objects
    Ntrain = 0  # no. training objects
    Ntest = 0  # no. test objects
    # Sparsity handling
    sparse = False
    observed_idxs = []
    full_N = 0
    # The data from the crowd
    C = None
    Ctest = None # data for the test points (excluding training)
    goldlabels = None
    # Indices into the current data set
    trainidxs = None
    testidxs = None
    conf_mat_ind = []  # indices into the confusion matrices corresponding to the current set of crowd labels
    # the joint likelihood (interim value saved to reduce computation)
    lnpCT = None
      
# Model parameters and hyper-parameters -----------------------------------------------------------------------------
    #The model
    alpha0 = None
    clusteridxs_alpha0 = [] # use this if you want to use an alpha0 where each matrix is diags prior for diags group of agents. This
    # is and array of indicies that indicates which of the original alpha0 groups should be used for each agent.
    alpha0_length = 1 # can be either 1, K or nclusters
    alpha0_cluster = [] # copy of the alpha0 values for each cluster.
    nu0 = None    
    lnkappa = []
    nu = []
    lnPi = []
    alpha = []
    E_t = []
    E_t_sparse = []
    #hyper-hyper-parameters: the parameters for the hyper-prior over the hyper-parameters. These are only used if you
    # run optimize_hyperparams
    gam_scale_alpha = []  #Gamma distribution scale parameters 
    gam_shape_alpha = 1 #Gamma distribution shape parameters --> confidence in seed values
    gam_scale_nu = []
    gam_shape_nu = 100
    
    optimise_alpha0_diagonals = False # simplify optimisation by using diagonal alpha0 only
# Initialisation ---------------------------------------------------------------------------------------------------
    def __init__(self, nclasses=2, nscores=2, alpha0=None, nu0=None, K=1, uselowerbound=False, dh=None, use_ml=False):
        '''
        Independent Bayesian classifier combination. The object can be trained using a data handler object dh,
        the example in the main function below, and the unit tests for further examples). However,
        This may be useful if you want IBCC to load the data from CSV files. However, if you are passing the data
        to IBCC as numpy arrays, do not use dh, and set all the parameters of the constructor manually.
        
        The most important functions are the constructor, where hyperparameters are set, and combine_classifications(),
        where the IBCC model is trained in unsupervised/semi-supervised mode and returns a combined prediction.
        
        Parameters
        ----------
        
        nclasses : int
            number of target classes
        nclases : int
            number of scores that the workers/agents/annotators/labellers/base classifiers can assign. Often, this 
            is the same as nclasses, but may differe when annotators can provide, for example, an "I don't know" label.
        alpha0 : nclasses x nscores numpy array
            Hyperparameters for the confusion matrices. Typically this is set to a weak prior, e.g.
            alpha0=np.ones((nclasses, nscores)) + np.eye(nclasses)
        nu0 : nclasses numpy array
            Hyperparameters for class proportions. Each value represents a prior pseudo-count of a target class. Setting
            nu0 to large values will give a strong prior. Setting one value larger than the others will increase prior
            probability of that class. A typical default may be nu0=np.ones(nclasses) * 10
        K : int
            Number of workers/base classifiers/annotators
        uselowerbound : bool
            Flag that determines whether to use the lower bound on the log-marginal likelihood to check for convergence,
            or to check for convergence in the model parameters. Setting this to true is useful for debugging and is 
            more likely to detect convergence correctly, but may be more computationally costly.
        dh : ibccdata.DataHandler object
            Object for loading the data from CSV files.
    
        '''
        if dh != None:
            self.nclasses = dh.nclasses
            self.nscores = len(dh.scores)
            self.alpha0 = dh.alpha0
            self.nu0 = dh.nu0
            self.K = dh.K
            self.uselowerbound = dh.uselowerbound
        else:
            self.nclasses = nclasses
            self.nscores = nscores
            self.alpha0 = alpha0
            self.nu0 = nu0
            self.K = K
            self.uselowerbound = uselowerbound

        if self.nu0 is None:
            self.nu0 = np.ones(self.nclasses)

        if self.alpha0 is None:
            self.alpha0 = np.ones((self.nclasses, self.nscores))
            if self.nclasses <= self.nscores:
                self.alpha0[range(self.nclasses), range(self.nscores)] += 1

        # Ensure we have float arrays so we can do division with these parameters properly
        self.nu0 = np.array(self.nu0).astype(float)
        if self.nu0.ndim==1:
            self.nu0 = self.nu0.reshape((self.nclasses,1))
        elif self.nu0.shape[0]!=self.nclasses and self.nu0.shape[1]==self.nclasses:
            self.nu0 = self.nu0.T
            
        if len(self.alpha0.shape) == 3:
            self.alpha0_length = self.alpha0.shape[2]
        else:
            self.alpha0_length = 1

        self.use_ml = use_ml
        if use_ml:
            self.uselowerbound = False


    def _init_params(self, force_reset=False):
        '''
        Checks that parameters are intialized, but doesn't overwrite them if already set up.
        '''
        if self.verbose:
            logging.debug('Initialising parameters...Alpha0: ' + str(self.alpha0))
        #if alpha is already initialised, and no new agents, skip this
        if self.alpha == [] or self.alpha.shape[2] != self.K or force_reset:
            self._init_lnPi()
        if self.verbose:
            logging.debug('Nu0: ' + str(self.nu0))
        if self.nu ==[] or force_reset:
            self._init_lnkappa()


    def _init_lnkappa(self):
        self.nu = deepcopy(np.float64(self.nu0))
        sumNu = np.sum(self.nu)
        self.lnkappa = psi(self.nu) - psi(sumNu)


    def _init_lnPi(self):
        '''
        Always creates new self.alpha and self.lnPi objects and calculates self.alpha and self.lnPi values according to 
        either the prior, or where available, values of self.E_t from previous runs.
        '''
        self.alpha0 = self.alpha0.astype(float)
        # if we specify different alpha0 for some agents, we need to do so for all K agents. The last agent passed in 
        # will be duplicated for any missing agents.
        if np.any(self.clusteridxs_alpha0): # map from diags list of cluster IDs
            if not np.any(self.alpha0_cluster):
                self.alpha0_cluster = self.alpha0
                self.alpha0_length = self.alpha0_cluster.shape[2]
            self.alpha0 = self.alpha0_cluster[:, :, self.clusteridxs_alpha0]        
        else:
            if len(self.alpha0.shape)==2:
                self.alpha0  = self.alpha0[:,:,np.newaxis]
            
            if self.alpha0.shape[2] < self.K:
                # We have a new dataset with more agents than before -- create more priors.
                nnew = self.K - self.alpha0.shape[2]
                alpha0new = self.alpha0[:, :, 0]
                alpha0new = alpha0new[:, :, np.newaxis]
                alpha0new = np.repeat(alpha0new, nnew, axis=2)
                self.alpha0 = np.concatenate((self.alpha0, alpha0new), axis=2)
        # Make sure self.alpha is the right size as well. Values of self.alpha not important as we recalculate below
        self.alpha0 = self.alpha0[:, :, :self.K] # make this the right size if there are fewer classifiers than expected
        self.alpha = np.zeros((self.nclasses, self.nscores, self.K), dtype=np.float) + self.alpha0

        self.lnPi = np.zeros((self.nclasses, self.nscores, self.K))

        if not self.use_ml:
            self._expec_lnpi(self.use_ml) # calculate alpha from the initial/prior values only in the first iteration
        else:
            self.lnPi[:] = np.log(0.1 / float(self.nscores - 1))
            self.lnPi[np.arange(self.nclasses), np.arange(self.nclasses), :] = np.log(0.9)


    def _init_t(self):
        if np.any(self.E_t):
            if self.sparse:
                oldE_t = self.E_t_sparse
            else:
                oldE_t = self.E_t
            if np.any(oldE_t):
                Nold = oldE_t.shape[0]
                if Nold > self.N:
                    Nold = self.N
        else:
            oldE_t = []

        # initialise t to the vote distributions
        self.E_t = np.zeros((self.N, self.nclasses)) + self.nu0.T
        for l in range(self.nclasses):
            self.E_t[:, l] += np.sum(self.C[l], axis=1)
        self.E_t /= np.sum(self.E_t, axis=1)[:, None]

        if np.any(oldE_t):
            self.E_t[0:Nold, :] = oldE_t[0:Nold, :]
        uncert_trainidxs = self.trainidxs.copy()  # look for labels that are not discrete values of valid classes
        for j in range(self.nclasses):
            # training labels
            row = np.zeros((1, self.nclasses))
            row[0, j] = 1
            jidxs = self.goldlabels == j
            uncert_trainidxs[jidxs] = False
            self.E_t[jidxs, :] = row
        # deal with uncertain training idxs
        for j in range(self.nclasses):
            # values a fraction above class j
            partly_j_idxs = np.bitwise_and(self.goldlabels[uncert_trainidxs] > j, self.goldlabels[uncert_trainidxs] < j + 1)
            partly_j_idxs = uncert_trainidxs[partly_j_idxs]
            self.E_t[partly_j_idxs, j] = (j + 1) - self.goldlabels[partly_j_idxs]
            # values a fraction below class j
            partly_j_idxs = np.bitwise_and(self.goldlabels[uncert_trainidxs] < j, self.goldlabels[uncert_trainidxs] > j - 1)
            partly_j_idxs = uncert_trainidxs[partly_j_idxs]
            self.E_t[partly_j_idxs, j] = self.goldlabels[partly_j_idxs] - j + 1
        if self.sparse:
            self.E_t_sparse = self.E_t # current working version is a sparse set of observations of the complete space of data points            


# Data preprocessing and helper functions --------------------------------------------------------------------------
    def _desparsify_crowdlabels(self, crowdlabels):
        '''
        Converts the IDs of data points in the crowdlabels to a set of consecutive integer indexes. If diags data point has
        no crowdlabels when using table format, it will be skipped.   
        '''
        if self.table_format_flag:
            # First, record which objects were actually observed.
            self.observed_idxs = np.argwhere(np.sum(np.isfinite(crowdlabels), axis=1) > 0).reshape(-1)
            # full set of test points will include those with crowd labels = NaN, unless gold labels are passed in 
            self.full_N = crowdlabels.shape[0]             
            if crowdlabels.shape[0] > len(self.observed_idxs):
                self.sparse = True
                # cut out the unobserved data points. We'll put them back in at the end of the classification procedure.
                crowdlabels = crowdlabels[self.observed_idxs, :]
        else:
            crowdobjects = crowdlabels[:,1].astype(int)
            self.observed_idxs, mappedidxs = np.unique(crowdobjects, return_inverse=True)
            self.full_N = int(np.max(crowdlabels[:,1])) + 1 # have to add one since indexes start from 0
            if self.full_N > len(self.observed_idxs):
                self.sparse = True
                # map the IDs so we skip unobserved data points. We'll map back at the end of the classification procedure.
                crowdlabels[:, 1] = mappedidxs
        return crowdlabels


    def _preprocess_goldlabels(self, goldlabels):
        if np.any(goldlabels) and self.sparse:
            if self.full_N<len(goldlabels):
                # the full set of test points that we output will come from the gold labels if longer than crowd labels
                self.full_N = len(goldlabels)
            goldlabels = goldlabels[self.observed_idxs]
            
        # Find out how much training data and how many total data points we have
        if np.any(goldlabels):
            len_t = goldlabels.shape[0]
            goldlabels[np.isnan(goldlabels)] = -1
        else:
            len_t = 0  # length of the training vector
        len_c = len(self.observed_idxs)# length of the crowdlabels
        # How many data points in total?
        if len_c > len_t:
            self.N = len_c
        else:
            self.N = len_t        
             
        # Make sure that goldlabels is the right size in case we have passed in a training set for the first idxs
        if not np.any(goldlabels):
            self.goldlabels = np.zeros(self.N) - 1
        elif goldlabels.shape[0] < self.N:
            extension = np.zeros(self.N - goldlabels.shape[0]) - 1
            self.goldlabels = np.concatenate((goldlabels, extension))
        else:
            self.goldlabels = goldlabels


    def _set_test_and_train_idxs(self, testidxs=None):
        # record the test and train idxs
        self.trainidxs = self.goldlabels > -1
        self.Ntrain = np.sum(self.trainidxs)
        # self.testidxs only includes points with crowd labels!
        if testidxs is not None:  # include the pre-specified set of unlabelled data points in the inference process. All
            # other data points are either training data or ignored.
            if self.full_N < len(testidxs): 
                self.full_N = len(testidxs)
            self.testidxs = testidxs[self.observed_idxs]
        else:  # If the test indexes are not specified explicitly, assume that all data points with a NaN or a -1 in the
            # training data must be test indexes.
            self.testidxs = np.bitwise_or(np.isnan(self.goldlabels), self.goldlabels < 0)
        self.testidxs = self.testidxs > 0
        self.Ntest = np.sum(self.testidxs)

        if self.Ntest == self.N:
            # we test on all indexes, so avoid using testidxs
            self.testidxs = None


    def _preprocess_crowdlabels(self, crowdlabels):
        # Initialise all objects relating to the crowd labels.
        C = {}
        crowdlabels[np.isnan(crowdlabels)] = -1
        if self.discretedecisions:
            crowdlabels = np.round(crowdlabels).astype(int)
        if self.table_format_flag:# crowd labels as a full KxN table? If false, use diags sparse 3-column list, where 1st
            # column=classifier ID, 2nd column = obj ID, 3rd column = score.
            self.K = crowdlabels.shape[1]
            for l in range(self.nscores):
                Cl = np.zeros((self.N, self.K))
                #crowd labels may not be supplied for all N data points in the gold labels, so use argwhere
                lidxs = np.argwhere(crowdlabels==l)
                Cl[lidxs[:,0], lidxs[:,1]] = 1
                if not self.discretedecisions:
                    if l + 1 < self.nscores:
                        partly_l_idxs = np.bitwise_and(crowdlabels > l, crowdlabels < (l+1))  # partly above l
                        Cl[partly_l_idxs] = (l + 1) - crowdlabels[partly_l_idxs]
                    if l > 0:
                        partly_l_idxs = np.bitwise_and(crowdlabels < l, crowdlabels > (l-1))  # partly below l
                        Cl[partly_l_idxs] = crowdlabels[partly_l_idxs] - l + 1
                C[l] = Cl
        else:
            if self.K < int(np.nanmax(crowdlabels[:,0]))+1:
                self.K = int(np.nanmax(crowdlabels[:,0]))+1 # add one because indexes start from 0
            for l in range(self.nscores):
                lIdxs = np.argwhere(crowdlabels[:, 2] == l)[:,0]
                data = np.ones((len(lIdxs), 1)).reshape(-1)
                rows = np.array(crowdlabels[lIdxs, 1]).reshape(-1)
                cols = np.array(crowdlabels[lIdxs, 0]).reshape(-1)
                
                if not self.discretedecisions:
                    partly_l_idxs = np.bitwise_and(crowdlabels[:, 2] > l, crowdlabels[:, 2] < l + 1)  # partly above l
                    data = np.concatenate((data, (l + 1) - crowdlabels[partly_l_idxs, 2]))
                    rows = np.concatenate((rows, crowdlabels[partly_l_idxs, 1].reshape(-1)))
                    cols = np.concatenate((cols, crowdlabels[partly_l_idxs, 0].reshape(-1)))
                    
                    partly_l_idxs = np.bitwise_and(crowdlabels[:, 2] < l, crowdlabels[:, 2] > l - 1)  # partly below l
                    data = np.concatenate((data, crowdlabels[partly_l_idxs, 2] - l + 1))
                    rows = np.concatenate((rows, crowdlabels[partly_l_idxs, 1].reshape(-1)))
                    cols = np.concatenate((cols, crowdlabels[partly_l_idxs, 0].reshape(-1)))
                Cl = csr_matrix(coo_matrix((data,(rows,cols)), shape=(self.N, self.K)))
                C[l] = Cl
        # Set and reset object properties for the new dataset
        self.C = C
        self.lnpCT = np.zeros((self.N, self.nclasses))
        self.conf_mat_ind = []
        # pre-compute the indices into the pi arrays
        # repeat for test labels only
        
        self.Ctest = {}

        for l in range(self.nscores):
            if self.testidxs is not None:
                self.Ctest[l] = C[l][self.testidxs, :]
            else:
                self.Ctest[l] = C[l]

        # Reset the pre-calculated data for the training set in case goldlabels has changed
        self.alpha_tr = None


    def _resparsify_t(self):
        '''
        Puts the expectations of target values, E_t, at the points we observed crowd labels back to their original 
        indexes in the output array. Values are inserted for the unobserved indices using only kappa (class proportions).
        '''
        E_t_full = np.zeros((self.full_N, self.nclasses))
        E_t_full[:] = (np.exp(self.lnkappa) / np.sum(np.exp(self.lnkappa),axis=0)).T
        E_t_full[self.observed_idxs,:] = self.E_t
        self.E_t_sparse = self.E_t  # save the sparse version
        self.E_t = E_t_full


# Run the inference algorithm --------------------------------------------------------------------------------------
    def combine_classifications(self, crowdlabels, goldlabels=None, testidxs=None, optimise_hyperparams=0, maxiter=200, 
                                table_format=False):
        '''
        Takes crowdlabels in either sparse list or table formats, along with optional training labels (goldlabels)
        and applies data-preprocessing steps before running inference for the model parameters and target labels.
        Returns the expected values of the target labels.
        
        Parameters
        ----------
        
        crowdlabels : N_labels x 3 numpy array or N_data_points x N_workers numpy array
            Where N_labels is the number of labels received from the crowd/base classifiers/all annotators; 
            N_data_points is the number of data points to be classified; and N_workers is the number of workers/
            annotators/base classifiers that have provided labels.
            
            The N_labels x 3 array is a sparse format and is more compact if most data points are 
            only labelled by a small subset of workers. Each row represents a label, where the first column is the 
            worker/agent/base classifier ID, the second column is the data point ID, and the third column is the label
            that was assigned by that worker to that data point. 
            
            The N_data_points x N_workers array is a matrix where each row corresponds to a data point and each column
            to a worker/agent/base classifier. Any missing entries should be np.NaN or -1. To use this matrix as 
            input, set table_format=True
        goldlabels : N_data_points numpy array
            Optional array for supplying any known training data. Missing labels, i.e. test locations, should have 
            values of -1.
        testidxs : N_data_points numpy array
            Optional array of boolean values indicating which indices require predictions. If not set, all idxs will be
            predicted. 
        optimise_hyperparams : bool
            Optimise nu0 and alpha0 using the Nelder-Mead algorithm.
        max_iter : int
            maximum number of iterations permitted
        table_format : bool
            Set this to true if the crowdlabels are a matrix with rows corresponding to data points and columns 
            corresponding to workers. 
        
        Returns
        -------
        
        E_t : N_data_points x nclasses numpy array
            Posterior class probabilities (expected t-values) for each data point. Each column corresponds to a class.
        
        '''
        self.table_format_flag = table_format
        oldK = self.K
        crowdlabels = self._desparsify_crowdlabels(crowdlabels)
        self._preprocess_goldlabels(goldlabels)        
        self._set_test_and_train_idxs(testidxs)
        self._preprocess_crowdlabels(crowdlabels)
        self._init_t()

        #Check that we have the right number of agents/base classifiers, K, and initialise parameters if necessary
        if self.K != oldK or not np.any(self.nu) or not np.any(self.alpha):  # data shape has changed or not initialised yet
            self._init_params()  

        # Either run the model optimisation or just use the inference method with fixed hyper-parameters  
        if optimise_hyperparams==1 or optimise_hyperparams=='ML':
            self.optimize_hyperparams(maxiter=maxiter)
        elif optimise_hyperparams==2 or optimise_hyperparams=='MAP':
            self.optimize_hyperparams(maxiter=maxiter, use_MAP=True)
        else:
            self._run_inference() 
        if self.sparse:
            self._resparsify_t()
        return self.E_t      


    def _convergence_measure(self, oldET):
        return np.max(np.abs(oldET - self.E_t))


    def _convergence_check(self):
        return (self.nIts>=self.max_iterations or self.change<self.conv_threshold) and self.nIts>self.min_iterations        


    def _run_inference(self):   
        '''
        Variational approximate inference. Assumes that all data and hyper-parameters are ready for use. Overwrite
        do implement EP or Gibbs' sampling etc.
        '''
        logging.info('IBCC: combining %i training points + %i noisy-labelled points' % (np.sum(self.trainidxs), 
                                                                                        len(self.observed_idxs)))
        oldL = -np.inf
        converged = False
        self.nIts = 0 #object state so we can check it later
        while not converged and self.keeprunning:
            oldET = self.E_t.copy()

            #update params
            self._expec_lnkappa(self.use_ml)
            if np.any(self.E_t):
                self._post_alpha()
            self._expec_lnpi(self.use_ml)

            self._expec_t()

            #check convergence every x iterations
            if np.mod(self.nIts, self.conv_check_freq) == self.conv_check_freq - 1:
                if self.uselowerbound:
                    L = self.lowerbound()
                    if self.verbose:
                        logging.debug('Lower bound: ' + str(L) + ', increased by ' + str(L - oldL))
                    self.change = (L - oldL) / np.abs(L)
                    oldL = L
                    if self.change < - self.conv_threshold * np.abs(L) and self.verbose:                
                        logging.warning('IBCC iteration %i absolute change was %s. Possible bug or rounding error?' 
                                        % (self.nIts, self.change))                    
                else:
                    self.change = self._convergence_measure(oldET)
                if self._convergence_check():
                    converged = True            
                elif self.verbose:
                    logging.debug('IBCC iteration %i absolute change was %s' % (self.nIts, self.change))
                    
            self.nIts+=1
            
        logging.info('IBCC finished in %i iterations (max iterations allowed = %i).' % (self.nIts, self.max_iterations))


# Posterior Updates to Hyperparameters --------------------------------------------------------------------------------
    def _post_alpha(self):  # Posterior Hyperparams
        # Save the counts from the training data so we only recalculate the test data on every iteration
        if self.alpha_tr is None:
            self.alpha_tr = np.zeros(self.alpha.shape)
            if self.Ntrain:
                for j in range(self.nclasses):
                    for l in range(self.nscores):
                        Tj = self.E_t[self.trainidxs, j].reshape((self.Ntrain, 1))
                        self.alpha_tr[j,l,:] = self.C[l][self.trainidxs,:].T.dot(Tj).reshape(-1)
            self.alpha_tr += self.alpha0
        # Add the counts from the test data
        for j in range(self.nclasses):
            for l in range(self.nscores):

                if self.testidxs is not None:
                    Tj = self.E_t[self.testidxs, j].reshape((self.Ntest, 1))
                else:
                    Tj = self.E_t[:, j].reshape((self.Ntest, 1))

                counts = self.Ctest[l].T.dot(Tj).reshape(-1)
                self.alpha[j, l, :] = self.alpha_tr[j, l, :] + counts


# Expectations: methods for calculating expectations with respect to parameters for the VB algorithm ------------------
    def _expec_lnkappa(self, use_ml=False):
        if use_ml:
            nu = np.reshape(np.sum(self.E_t, 0), self.nu0.shape)  # ignores nu0
            lnkappa = np.log((nu - 1) / float(np.sum(nu, 0) - nu.shape[0]))
        else:
            nu = self.nu0 + np.reshape(np.sum(self.E_t, 0), self.nu0.shape)
            lnkappa = (psi(nu) - psi(np.sum(nu, 0)))
        self.nu = nu
        self.lnkappa = lnkappa


    def _expec_lnpi(self, use_ml=False):
        alpha_sum = np.sum(self.alpha, 1)
        for s in range(self.nscores):

            if use_ml:
                alpha_sum -= self.alpha.shape[1]
                self.lnPi[:, s, :] = np.log((self.alpha[:, s, :] - 1) / float(alpha_sum))  # mode of a dirichlet
            else:
                self.lnPi[:, s, :] = psi(self.alpha[:, s, :]) - psi(alpha_sum)


    def _expec_t(self):
        self._lnjoint()
        joint = self.lnpCT
        if self.testidxs is not None:
            joint = joint[self.testidxs, :]
        else:
            joint = np.copy(joint)

        # ensure that the values are not too small
        joint -= np.max(joint, 1)[:, np.newaxis]
        joint = np.exp(joint)
        norma = np.sum(joint, axis=1)[:, np.newaxis]
        pT = joint / norma

        # update targets
        if self.testidxs is not None:
            self.E_t[self.testidxs, :] = pT
        else:
            self.E_t = pT
   
# Likelihoods of observations and current estimates of parameters --------------------------------------------------
    def _lnjoint(self, alldata=False):
        '''
        For use with crowdsourced data in table format (should be converted on input)
        '''
        if self.uselowerbound or alldata:
            for j in range(self.nclasses):
                data = None
                for l in range(self.nscores):
                    if self.table_format_flag:
                        data_l = self.C[l] * self.lnPi[j, l, :][np.newaxis,:]
                    else:
                        data_l = self.C[l].multiply(self.lnPi[j, l, :][np.newaxis,:])

                    data_l = np.array(data_l.sum(axis=1)).reshape(-1)

                    data = data_l if data is None else data+data_l

                self.lnpCT[:, j] = data + self.lnkappa[j]
        else:  # no need to calculate in full
            for j in range(self.nclasses):
                data = None
                for l in range(self.nscores):
                    if self.table_format_flag:
                        data_l = self.Ctest[l] * self.lnPi[j, l, :][np.newaxis,:]
                    else:
                        data_l = self.Ctest[l].multiply(self.lnPi[j, l, :][np.newaxis,:])

                    data_l = np.array(data_l.sum(axis=1)).reshape(-1)

                    data = data_l if data is None else data+data_l

                if self.testidxs is not None:
                    self.lnpCT[self.testidxs, j] = data + self.lnkappa[j]
                else:
                    self.lnpCT[:, j] = np.array(data).reshape(-1) + self.lnkappa[j]
        
    def _post_lnkappa(self):
        lnpKappa = gammaln(np.sum(self.nu0)) - np.sum(gammaln(self.nu0)) + sum((self.nu0 - 1) * self.lnkappa)
        return lnpKappa
        
    def _q_lnkappa(self):
        lnqKappa = gammaln(np.sum(self.nu)) - np.sum(gammaln(self.nu)) + np.sum((self.nu - 1) * self.lnkappa)
        return lnqKappa

    def _q_ln_t(self):
        ET = self.E_t[self.E_t != 0]
        return np.sum(ET * np.log(ET))         

    def _post_lnpi(self):
        x = np.sum((self.alpha0-1) * self.lnPi,1)
        z = gammaln(np.sum(self.alpha0,1)) - np.sum(gammaln(self.alpha0),1)
        return np.sum(x+z)
                    
    def _q_lnPi(self):
        x = np.sum((self.alpha-1) * self.lnPi,1)
        z = gammaln(np.sum(self.alpha,1)) - np.sum(gammaln(self.alpha),1)
        return np.sum(x+z)
# Lower Bound ---------------------------------------------------------------------------------------------------------       
    def lowerbound(self):
        # Expected Energy: entropy given the current parameter expectations
        lnpCT = self._post_lnjoint_ct()
        lnpPi = self._post_lnpi()
        lnpKappa = self._post_lnkappa()
        EEnergy = lnpCT + lnpPi + lnpKappa
        
        # Entropy of the variational distribution
        lnqT = self._q_ln_t()
        lnqPi = self._q_lnPi()
        lnqKappa = self._q_lnkappa()
        H = lnqT + lnqPi + lnqKappa
        
        # Lower Bound
        L = EEnergy - H
        if self.verbose:
            logging.debug('EEnergy %.3f, H %.3f, lnpCT %.3f, lnqT %.3f, lnpKappa %.3f, lnqKappa %.3f, lnpPi %.3f, lnqPi %.3f' % \
                      (EEnergy, H, lnpCT, lnqT, lnpKappa, lnqKappa, lnpPi, lnqPi))
            logging.debug('lnpCT-lnqT = %.4f. lnpKappa-lnqKappa = %.4f. lnpPi-lnqPi = %.4f' % (lnpCT-lnqT, 
                                                                                   lnpKappa - lnqKappa, lnpPi - lnqPi))
        return L
# Hyperparameter Optimisation ------------------------------------------------------------------------------------------
    def _set_hyperparams(self,hyperparams):
        n_alpha_elements = len(hyperparams) - 1 #self.nclasses -- no longer optimising nu0, only its scale factor
        alpha_shape = (self.nclasses, self.nscores, self.alpha0_length)
        
        if self.optimise_alpha0_diagonals:
            alpha0 = np.ones((self.nclasses, self.nscores)) * hyperparams[self.nclasses:self.nclasses*2][:, None]
            alpha0[np.arange(self.nclasses), np.arange(self.nclasses)] += hyperparams[:self.nclasses] # add to the diagonals

        else:
            alpha0 = hyperparams[0:n_alpha_elements].reshape(alpha_shape)

        # nu0 = np.array(hyperparams[-self.nclasses:]).reshape(self.nclasses, 1)
        nu0 = np.ones(self.nclasses) * hyperparams[-1]

        if self.clusteridxs_alpha0 != []:
            self.alpha0_cluster = alpha0
        else:
            self.alpha0 = alpha0  
        self.nu0 = nu0
        return alpha0, nu0

    def _get_hyperparams(self):
        if self.clusteridxs_alpha0 != []:
            alpha0 = self.alpha0_cluster

        elif self.optimise_alpha0_diagonals:
            alpha0 = self.alpha0
            alpha0_diags = alpha0[range(self.nclasses), range(self.nclasses)]
            off_diag_idxs = np.mod(np.arange(self.nclasses) + 1, self.nclasses)
            alpha0_scale = alpha0[range(self.nclasses), off_diag_idxs]
            alpha0 = np.append(alpha0_diags, alpha0_scale)
        else:
            alpha0 = self.alpha0

        # only the scaling of nu0 is optimised to prevent overfitting
        return np.concatenate((alpha0.flatten(), self.nu0[0]))
    
    def _post_lnjoint_ct(self):
        # If we have not already calculated lnpCT for the lower bound, then make sure we recalculate using all data
        if not self.uselowerbound:
            self._lnjoint(alldata=True)
        if self.sparse:
            lnpCT = np.sum(self.E_t_sparse * self.lnpCT)            
        else:
            lnpCT = np.sum(self.E_t * self.lnpCT)
        return lnpCT
                
    def ln_modelprior(self):
        #Check and initialise the hyper-hyper-parameters if necessary
        if self.gam_scale_alpha==[] or (len(self.gam_scale_alpha.shape) == 3 and self.gam_scale_alpha.shape[2]!=self.alpha0.shape[2]):
            self.gam_shape_alpha = np.float(self.gam_shape_alpha)
            # if the scale was not set, assume current values of alpha0 are the means given by the hyper-prior
            self.gam_scale_alpha = self.alpha0/self.gam_shape_alpha
        if self.gam_scale_nu==[]:
            self.gam_shape_nu = np.float(self.gam_shape_nu)
            # if the scale was not set, assume current values of nu0 are the means given by the hyper-prior
            self.gam_scale_nu = self.nu0/self.gam_shape_nu
        
        #Gamma distribution over each value. Set the parameters of the gammas.
        p_alpha0 = gamma.logpdf(self.alpha0, a=self.gam_shape_alpha, scale=self.gam_scale_alpha)
        p_nu0 = gamma.logpdf(self.nu0, a=self.gam_shape_nu, scale=self.gam_scale_nu)
        
        return np.sum(p_alpha0) + np.sum(p_nu0)

    def neg_marginal_likelihood(self, hyperparams, use_MAP):
        '''
        Weight the marginal log data likelihood by the hyper-prior. Unnormalised posterior over the hyper-parameters.
        '''
        if self.verbose:
            logging.debug("Hyper-parameters: %s" % str(hyperparams))
        if np.any(np.isnan(hyperparams)):
            return np.inf
        hyperparams = self._set_hyperparams(hyperparams)
        if np.any(hyperparams[0] <=0 ) or np.any(hyperparams[1] <= 0 ):
            return np.inf
        
        #ensure new alpha0 and nu0 values are used when updating E_t
        self._init_params(force_reset=True)
        #run inference algorithm
        self._run_inference() 
        
        #calculate likelihood from the fitted model
        data_loglikelihood = self.lowerbound()
        lml = data_loglikelihood 
        
        if use_MAP:
            log_model_prior = self.ln_modelprior() 
            lml += log_model_prior
            logging.debug("Log marginal likelihood/model evidence + log model hyperprior: %f" % lml)
        else:
            logging.debug("Log marginal likelihood/model evidence: %f" % lml)
        return -lml #returns Negative!
    
    def optimize_hyperparams(self, maxiter=20, use_MAP=False):
        ''' 
        Assuming gamma distributions over the hyper-parameters, we find the MAP values. The combiner object is updated
        to contain the optimal values, searched for using BFGS.
        '''
        #Evaluate the first guess using the current value of the hyper-parameters
        initialguess = self._get_hyperparams()
        initial_nlml = self.neg_marginal_likelihood(initialguess, use_MAP)
        ftol = 1.0 #1e-3 #np.abs(initial_nlml / 1e3)
        #ftol = np.abs(np.log(self.heatGP[1].ls[0]) * 1e-4) 
        self.conv_threshold = ftol / 10.0
        
        #opt_hyperparams = fmin_cobyla(self.neg_marginal_likelihood, initialguess, constraints, maxfun=maxiter, rhobeg=rhobeg, rhoend=rhoend)
        opt_hyperparams, _, _, _, _ = fmin(self.neg_marginal_likelihood, initialguess, args=(use_MAP,), maxfun=maxiter,
                                                     full_output=True, ftol=ftol, xtol=1e100)

        opt_hyperparams = self._set_hyperparams(opt_hyperparams)
        msg = "Optimal hyper-parameters: "
        for param in opt_hyperparams:
            if not np.isscalar(param):
                for val in param:
                    msg += str(val) + ', '
            else:
                msg += str(param) + ', '
        logging.debug(msg)
        
        return self.E_t

