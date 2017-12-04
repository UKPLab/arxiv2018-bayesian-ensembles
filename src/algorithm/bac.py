'''
Error analysis -- cases include:
- Assuming an annotation provided by only one base classifier ( p(O | B or I ) = high for other base classifiers)
- Switching from I-O-O-O to I-O-B-I -- hard to see why this would occur unless transition matrix contains
 p(B | prev_I) < p(B | prev_O) and confusion matrices contain p( I | B ) > p(I | O). 
 Another cause: p( c=O | c_prev = O, t=B) = high (missing an annotation is quite common), 
 p( c=O | c_prev = I, t=B) = low (missing a new annotation straight after another is rare). 
 This error occurs when
 some annotators continue the annotation for longer than others; BAC decides there are two annotations,
 one that agrees with the other combination methods, and one that contains the tail of the overly long
 annotations.  

Possible solution to these two cases is to reduce the prior on missing an annotation, i.e. p( c=O | c_next=whatever, t=B)?
reduce alpha0[2, 1, 1, :], reduce alpha0[2,0,0,:], perhaps increase alpha0[2,2,:,:] to compensate?

Example error:
-- line 2600, argmin binary experiment
-- Some annotators start the annotation early
-- The annotation is split because the B labels are well trusted
-- But the early annotation is marked because there is a high chance of missing annotations by some annotators.
-- Better model might assume lower chance of I-B transition + lower change of missing something?


Created on Jan 28, 2017

@author: Melvin Laux
'''

import numpy as np
from scipy.misc import logsumexp
from scipy.special import psi
from scipy.special import gammaln
from scipy.optimize.optimize import fmin
from numpy import argmax
from joblib import Parallel, delayed

class BAC(object):
    '''
    classdocs
    '''
    
    K = None  # number of annotators
    L = None  # number of class labels
    
    nscores = None  # number of possible values a token can have, usually this is L + 1 (add one to account for unannotated tokens)
    
    nu0 = None  # ground truth priors
    
    lnA = None  # transition matrix
    lnPi = None  # worker confusion matrices
    
    q_t = None  # current true label estimates
    q_t_old = None  # previous true labels estimates
    
    iter = 0  # current iteration
    
    max_iter = None  # maximum number of iterations
    eps = None  # maximum difference of estimate differences in convergence chack
    

    def __init__(self, L=3, K=5, max_iter=100, eps=1e-5, inside_labels=[0], outside_labels=[1, -1], before_doc_idx=-1,
                 exclusions=None, alpha0=None, nu0=None):
        '''
        Constructor
        '''
        # initialise variables
        self.L = L
        self.nscores = L
        self.K = K
        
        # set priors
        if alpha0 is None:
            # dims: true_label[t], current_annoc[t],  previous_anno c[t-1], annotator k
            self.alpha0 = np.ones((self.L, self.nscores, self.nscores + 1, self.K)) + 1.0 * np.eye(self.L)[:, :, None, None] 
        else:
            self.alpha0 = alpha0
    
        if nu0 is None:
            self.nu0 = np.ones((L + 1, L))
        else:
            self.nu0 = nu0
        
        beginning_labels = np.ones(self.nscores + 1, dtype=bool)
        beginning_labels[inside_labels] = False
        beginning_labels[outside_labels] = False
                
        # set priors for invalid transitions (to low values)
        for inside_label in inside_labels:
            # pseudo-counts for the transitions that are not allowed from outside to inside
            #disallowed_count = self.alpha0[:, inside_label, outside_labels, :] - 1 # pseudocount is (alpha0 - 1)
            # split the disallowed count between the possible beginning labels
            #self.alpha0[:, beginning_labels[:self.nscores], outside_labels, :] += disallowed_count / np.sum(beginning_labels)
            # set the disallowed transition to as close to zero as possible
            self.alpha0[:, inside_label, outside_labels, :] = np.nextafter(0, 1)
            self.nu0[outside_labels, inside_label] = np.nextafter(0, 1)
        # self.nu0[1, 1] += 1.0
        
        if exclusions is not None:
            for label, excluded in dict(exclusions).iteritems():
                self.alpha0[:, excluded, label, :] = np.nextafter(0, 1)
                self.nu0[label, excluded] = np.nextafter(0, 1)
            
        # initialise transition and confusion matrices
        self._initA()
        self._init_lnPi()
        
        self.before_doc_idx = before_doc_idx  # identifies which true class value is assumed for the label before the start of a document
        
        self.max_iter = max_iter  # maximum number of iterations
        self.eps = eps  # threshold for convergence 
        
        self.verbose = False  # can change this if you want progress updates to be printed
        
    def _initA(self):
        #self.lnA = np.log(self.nu0 / np.sum(self.nu0, 1)[:, None])
        self.lnA = psi(self.nu0) - psi(np.sum(self.nu0, -1))[:, None]

    def _init_lnPi(self):
        #self.lnPi = np.log(self.alpha0 / np.sum(self.alpha0, 1)[:, None, :, :])
        psi_alpha_sum = psi(np.sum(self.alpha0, 1))
        dims = self.alpha0.shape
        self.lnPi = np.zeros(dims)
        for s in range(dims[1]):  
            self.lnPi[:, s, :, :] = psi(self.alpha0[:, s, :, :]) - psi_alpha_sum        
        
    def lowerbound(self, C, doc_start):
        '''
        Compute the variational lower bound on the log marginal likelihood. 
        '''
        lnpC = 0
        C = C.astype(int)
        C_prev = np.concatenate((np.zeros((1, C.shape[1]), dtype=int), C[:-1, :]))
        C_prev[doc_start.flatten() == 1, :] = 0
        C_prev[C_prev == 0] = self.before_doc_idx + 1  # document starts or missing labels
        valid_labels = (C != 0).astype(float)
        
        # this doesn't seem quite right -- should include the transition matrix
        for j in range(self.L):
            lnpCj = valid_labels * self.lnPi[j, C - 1, C_prev - 1, np.arange(self.K)[None, :]] * self.q_t[:, j:j+1]
            lnpCj[np.isinf(lnpCj) | np.isnan(lnpCj)] = 0
            lnpC += lnpCj            
            
        lnpt = self.lnA[None, :, :] * self.q_t_joint
        lnpt[np.isinf(lnpt) | np.isnan(lnpt)] = 0
        
        lnpCt = np.sum(lnpC) + np.sum(lnpt)
                        
        lnqt = np.log(self.q_t_joint / np.sum(self.q_t_joint, axis=1)[:, None, :]) * self.q_t_joint
        lnqt[np.isinf(lnqt) | np.isnan(lnqt)] = 0
        lnqt = np.sum(lnqt)
            
        # E[ln p(\pi | \alpha_0)]
        x = np.sum((self.alpha0 - 1) * self.lnPi, 1)
        x[np.isinf(x)] = 0
        z = gammaln(np.sum(self.alpha0, 1)) - np.sum(gammaln(self.alpha0), 1)
        z[np.isinf(z)] = 0
        lnpPi = np.sum(x + z)
                    
        # E[ln q(\pi)]
        x = np.sum((self.alpha - 1) * self.lnPi, 1)
        x[np.isinf(x)] = 0
        z = gammaln(np.sum(self.alpha, 1)) - np.sum(gammaln(self.alpha), 1)
        z[np.isinf(z)] = 0
        lnqPi = np.sum(x + z)
        
        # E[ln p(A | nu_0)]
        x = np.sum((self.nu0 - 1) * self.lnA, 1)
        x[np.isinf(x)] = 0
        z = gammaln(np.sum(self.nu0, 1)) - np.sum(gammaln(self.nu0), 1)
        z[np.isinf(z)] = 0
        lnpA = np.sum(x + z) 
        
        # E[ln q(A)]
        x = np.sum((self.nu - 1) * self.lnA, 1)
        x[np.isinf(x)] = 0
        z = gammaln(np.sum(self.nu, 1)) - np.sum(gammaln(self.nu), 1)
        z[np.isinf(z)] = 0
        lnqA = np.sum(x + z)
        
        print 'Computing LB: %f, %f, %f, %f, %f, %f' % (lnpCt, lnqt, lnpPi, lnqPi, lnpA, lnqA)
        lb = lnpCt - lnqt + lnpPi - lnqPi + lnpA - lnqA        
        return lb
        
    def optimize(self, C, doc_start, maxfun=50):
        ''' 
        Run with MLII optimisation over the lower bound on the log-marginal likelihood. Optimizes the diagonals of the
        confusion matrix prior (assuming a single value for all diagonals) and the scaling of the transition matrix 
        hyperparameters.
        '''
        
        def neg_marginal_likelihood(hyperparams, C, doc_start):
            # set hyperparameters
            diagidxs = np.arange(self.L)
            self.alpha0[diagidxs, diagidxs, :, :] = np.exp(hyperparams[0]) * np.ones(self.L)[:, None, None]
            
            self.nu0 = np.ones((self.L + 1, self.L)) * np.exp(hyperparams[1])
            for inside_label in self.inside_labels:
                self.nu0[[1, -1], inside_label] = np.nextafter(0, 1)
                        
            self._initA()
            self._init_lnPi()
                        
            # run the method
            self.run(C, doc_start)
            
            # compute lower bound
            lb = self.lowerbound(C, doc_start)
            
            print "Lower bound: %.5f, alpha_0 diagonals = %.3f, nu0 scale = %.3f" % (lb, np.exp(hyperparams[0]),
                                                                                         np.exp(hyperparams[1]))
            
            return -lb
        
        initialguess = [np.log(self.alpha0[0, 0, 0, 0]), np.log(self.nu0[0, 0])]
        ftol = 1e-3
        opt_hyperparams, _, _, _, _ = fmin(neg_marginal_likelihood, initialguess, args=(C, doc_start), maxfun=maxfun,
                                                     full_output=True, ftol=ftol, xtol=1e100) 
            
        print "Optimal hyper-parameters: alpha_0 diagonals = %.3f, nu0 scale = %.3f" % (np.exp(opt_hyperparams[0]),
                                                                                        np.exp(opt_hyperparams[1]))

        return self.most_probable_sequence(C, doc_start)
            
    def run(self, C, doc_start):
        '''
        Runs the BAC algorithm with the given annotations and list of document starts.
        '''
        
        # validate input data
        assert C.shape[0] == doc_start.shape[0]
        
        # transform input data to desired format: unannotated tokens represented as zeros
        C = C.astype(int) + 1
        doc_start = doc_start.astype(bool)
        
        # initialise variables
        self.iter = 0
        self.q_t_old = np.zeros((C.shape[0], self.L))
        self.q_t = np.ones((C.shape[0], self.L))
        
        oldlb = -np.inf
        
        # main inference loop
        while not self._converged():
            
            # print status if desired
            if self.verbose:
                print "BAC iteration %i in progress" % self.iter
            
            # calculate alphas and betas using forward-backward algorithm 
            r_ = _parallel_forward_pass(C, self.lnA[0:self.L, :], self.lnPi, self.lnA[self.before_doc_idx, :],
                                               doc_start, self.before_doc_idx)
            lambd = _parallel_backward_pass(C, self.lnA[0:self.L, :], self.lnPi, doc_start, self.before_doc_idx)
            
            # update q_t and q_t_joint
            self.q_t_old = self.q_t
            self.q_t_joint = _parallel_expec_joint_t(r_, lambd, self.lnA, self.lnPi, C, doc_start, self.before_doc_idx)
            self.q_t = np.sum(self.q_t_joint, axis=1)
            
            # update E_lnA
            self.lnA, self.nu = _calc_q_A(self.q_t_joint, self.nu0)

            # update E_lnpi
            self.alpha = _post_alpha(self.q_t, C, self.alpha0, doc_start, self.before_doc_idx)            
            self.lnPi = _calc_q_pi(self.alpha)
            
            lb = self.lowerbound(C, doc_start)
            print 'Lower bound = %.5f, diff = %.5f' % (lb, lb - oldlb)
            oldlb = lb
            
            # increase iteration number
            self.iter += 1            

        return self._most_probable_sequence(C, doc_start)
    
    def _most_probable_sequence(self, C, doc_start):
        '''
        Use Viterbi decoding to ensure we make a valid prediction. There
        are some cases where most probable sequence does not match argmax(self.q_t,1). E.g.:
        [[0.41, 0.4, 0.19], [[0.41, 0.42, 0.17]].
        Taking the most probable labels results in an illegal sequence of [0, 1]. 
        Using most probable sequence would result in [0, 0].
        '''
        
        lnV = np.zeros((C.shape[0], self.L))
        prev = np.zeros((C.shape[0], self.L))  # most likely previous states
                
        for t in range(0, C.shape[0]):
            for l in xrange(self.L):
                if doc_start[t]:
                    likelihood_current = np.sum((C[0, :] != 0) * self.lnPi[l, C[0, :] - 1, self.before_doc_idx,
                                                                         np.arange(self.K)])
                    lnV[t, l] = self.lnA[self.before_doc_idx, l] + likelihood_current
                else:
                    likelihood_current = np.sum((C[t, :] != 0) * self.lnPi[l, C[t, :] - 1, C[t - 1, :] - 1, np.arange(self.K)]) 
                    p_current = lnV[t - 1, :] + self.lnA[:self.L, l] + likelihood_current
                    lnV[t, l] = np.max(p_current)
                    prev[t, l] = np.argmax(p_current, axis=0)
            
        # decode
        seq = np.zeros(C.shape[0], dtype=int)
        pseq = np.zeros((C.shape[0], self.L), dtype=float)
        
        for t in xrange(C.shape[0] - 1, -1, -1):
            if t + 1 == C.shape[0] or doc_start[t + 1]:
                seq[t] = np.argmax(lnV[t, :])
            else:
                seq[t] = prev[t + 1, seq[t + 1]]
            pseq[t, :] = np.exp(lnV[t, :] - logsumexp(lnV[t, :]))
        
        return pseq, seq

    def _converged(self):
        '''
        Calculates whether the algorithm has _converged or the maximum number of iterations is reached.
        The algorithm has _converged when the maximum difference of an entry of q_t between two iterations is 
        smaller than the given epsilon.
        '''
        if self.verbose:
            print "Difference in values at iteration %i: %.5f" % (self.iter, np.max(np.abs(self.q_t_old - self.q_t)))
        return ((self.iter >= self.max_iter) or np.max(np.abs(self.q_t_old - self.q_t)) < self.eps)

def _calc_q_A(E_t, nu0):
    '''
    Update the transition model.
    '''
    nu = nu0 + np.sum(E_t, 0)
    q_A = psi(nu) - psi(np.sum(nu, -1))[:, None]
    
    if np.any(np.isnan(q_A)):
        print '_calc_q_A: nan value encountered!'
    
    return q_A, nu

def _calc_q_pi(alpha):
    '''
    Update the annotator models.
    '''
    psi_alpha_sum = psi(np.sum(alpha, 1))
    dims = alpha.shape
    q_pi = np.zeros(dims)
    for s in range(dims[1]): 
        q_pi[:, s, :, :] = psi(alpha[:, s, :, :]) - psi_alpha_sum
    
    return q_pi
    
def _post_alpha(E_t, C, alpha0, doc_start, before_doc_idx=-1):  # Posterior Hyperparameters
    '''
    Update alpha.
    '''
    
    dims = alpha0.shape
    
    alpha = alpha0.copy()
    
    prev_missing = np.concatenate((np.zeros((1, C.shape[1])), C[:-1, :] == 0), axis=0)
    doc_start = doc_start + prev_missing
    
    for j in xrange(dims[0]): 
        Tj = E_t[:, j] 
        
        for l in xrange(dims[1]): 
            counts = ((C == l + 1) * doc_start).T.dot(Tj).reshape(-1)    
            alpha[j, l, before_doc_idx, :] = alpha0[j, l, before_doc_idx, :] + counts
            
            for m in xrange(dims[1]): 
                counts = ((C == l + 1)[1:, :] * (1 - doc_start[1:]) * (C == m + 1)[:-1, :]).T.dot(Tj[1:]).reshape(-1)
                alpha[j, l, m, :] = alpha[j, l, m, :] + counts
    
    return alpha

def _expec_t(lnR_, lnLambda):
    '''
    Calculate label probabilities for each token.
    '''
    return np.exp(lnR_ + lnLambda - logsumexp(lnR_ + lnLambda, axis=1)[:, None]) 
    
def _expec_joint_t(lnR_, lnLambda, lnA, lnPi, C, doc_start, before_doc_idx=-1):
    '''
    Calculate joint label probabilities for each pair of tokens.
    '''  
    # initialise variables
    T = lnR_.shape[0]
    L = lnA.shape[-1]
    K = lnPi.shape[-1]
    
    lnS = np.repeat(lnLambda[:, None, :], L + 1, 1)    
    flags = -np.inf * np.ones_like(lnS)
    # iterate though everything, calculate joint expectations
    for t in xrange(T):
        
        if doc_start[t]:
            flags[t, before_doc_idx, :] = 0
            lnS[t, before_doc_idx, :] += np.dot(lnPi[:, C[t, :] - 1, before_doc_idx, np.arange(K)], C[t, :] != 0) \
                                                + lnA[before_doc_idx, :] 
            continue
        
        for l_prev in xrange(L):
            flags[t, l_prev, :] = 0 
            prev_idx = C[t - 1, :] - 1
            prev_idx[prev_idx == -1] = before_doc_idx            
            lnS[t, l_prev, :] += np.dot(lnPi[:, C[t, :] - 1, prev_idx, np.arange(K)], C[t, :] != 0) \
                                                + lnA[l_prev, :] + lnR_[t - 1, l_prev]  
    # normalise and return  
    lnS = lnS + flags 
    if np.any(np.isnan(np.exp(lnS))):
        print '_expec_joint_t: nan value encountered (1) '
        
    lnS = lnS - logsumexp(lnS, axis=(1, 2))[:, None, None]
        
    if np.any(np.isnan(np.exp(lnS))):
        print '_expec_joint_t: nan value encountered' 
            
    return np.exp(lnS)

def _parallel_expec_joint_t(lnR_, lnLambda, lnA, lnPi, C, doc_start, before_doc_idx=-1):
    '''
    Calculate joint label probabilities for each pair of tokens.
    '''  
    # initialise variables
    T = lnR_.shape[0]
    L = lnA.shape[-1]
    
    lnS = np.repeat(lnLambda[:, None, :], L + 1, 1)    
    flags = -np.inf * np.ones_like(lnS)
    
    flags[np.where(doc_start == 1)[0], before_doc_idx, :] = 0
    flags[np.where(doc_start == 0)[0], :, :] = 0
    flags[np.where(doc_start == 0)[0], before_doc_idx, :] = -np.inf
                                                
    res = Parallel(n_jobs=-1)(delayed(calc_joint_t)(i, lnR_, lnLambda, lnA, lnPi, C, doc_start, before_doc_idx) for i in range(T))
    # print res
      
    # normalise and return  
    lnS = np.array(res)  # + flags 
    if np.any(np.isnan(np.exp(lnS))):
        print '_expec_joint_t: nan value encountered (1) '
        
    lnS = lnS - logsumexp(lnS, axis=(1, 2))[:, None, None]
        
    if np.any(np.isnan(np.exp(lnS))):
        print '_expec_joint_t: nan value encountered' 
            
    return np.exp(lnS)

def calc_joint_t(t, lnR_, lnLambda, lnA, lnPi, C, doc_start, before_doc_idx):
    '''
    Calculates the joint probabilities for a single pair of tokens.
    '''
    # initialise variables
    L = lnA.shape[-1]
    K = lnPi.shape[-1]
    
    lnS = np.repeat(lnLambda[:, None, :], L + 1, 1)    
    flags = -np.inf * np.ones_like(lnS)
        
    if doc_start[t]:
        flags[t, before_doc_idx, :] = 0
        lnS[t, before_doc_idx, :] += np.dot(lnPi[:, C[t, :] - 1, before_doc_idx, np.arange(K)], C[t, :] != 0) \
                                                + lnA[before_doc_idx, :] 
            
        return lnS[t, :, :] + flags[t, :, :]
        
    for l_prev in xrange(L):
        flags[t, l_prev, :] = 0 
        prev_idx = C[t - 1, :] - 1
        prev_idx[prev_idx == -1] = before_doc_idx            
        lnS[t, l_prev, :] += np.dot(lnPi[:, C[t, :] - 1, prev_idx, np.arange(K)], C[t, :] != 0) \
                                            + lnA[l_prev, :] + lnR_[t - 1, l_prev]
    
    return lnS[t, :, :] + flags[t, :, :]

def _forward_pass(C, lnA, lnPi, initProbs, doc_start, before_doc_idx=-1, skip=True):
    '''
    Perform the forward pass of the Forward-Backward algorithm (for each document separately).
    '''
    T = C.shape[0]  # infer number of tokens
    L = lnA.shape[0]  # infer number of labels
    K = lnPi.shape[-1]  # infer number of annotators
    
    # initialise variables
    lnR_ = np.zeros((T, L))
    scaling_factor = np.zeros((T, 1))
    
    mask = np.ones_like(C)
    
    if skip:
        mask = (C != 0)
    
    # iterate through all tokens, starting at the beginning going forward
    for t in xrange(T):
        if doc_start[t]:
            lnR_[t, :] = initProbs + np.sum(np.dot(lnPi[:, C[t, :] - 1, before_doc_idx, np.arange(K)], mask[t, :][:, None]),
                                           axis=1)
        else:
            # iterate through all possible labels
            for l in xrange(L):            
                prev_idx = C[t - 1, :] - 1
                prev_idx[prev_idx == -1] = before_doc_idx
                lnR_[t, l] = logsumexp(lnR_[t - 1, :] + lnA[:, l]) + np.sum(mask[t, :] * lnPi[l, C[t, :] - 1, prev_idx,
                                                                                        np.arange(K)])
        
        # normalise
        scaling_factor[t, :] = logsumexp(lnR_[t, :])
        lnR_[t, :] = lnR_[t, :] - scaling_factor[t, :]
            
    return lnR_

def _doc_forward_pass(C, lnA, lnPi, initProbs, before_doc_idx=-1, skip=True):
    '''
    Perform the forward pass of the Forward-Backward algorithm (for a single document).
    '''
    T = C.shape[0]  # infer number of tokens
    L = lnA.shape[0]  # infer number of labels
    K = lnPi.shape[-1]  # infer number of annotators
    
    # initialise variables
    lnR_ = np.zeros((T, L))
    scaling_factor = np.zeros((T, 1))
    
    mask = np.ones_like(C)
    
    if skip:
        mask = (C != 0)
        
    lnR_[0, :] = initProbs + np.sum(np.dot(lnPi[:, C[0, :] - 1, before_doc_idx, np.arange(K)], mask[0, :][:, None]),
                                           axis=1)
    
    scaling_factor[0, :] = logsumexp(lnR_[0, :])
    lnR_[0, :] = lnR_[0, :] - scaling_factor[0, :]
    
    # iterate through all tokens, starting at the beginning going forward
    for t in xrange(1, T):
        # iterate through all possible labels
        for l in xrange(L):            
            prev_idx = C[t - 1, :] - 1
            prev_idx[prev_idx == -1] = before_doc_idx
            lnR_[t, l] = logsumexp(lnR_[t - 1, :] + lnA[:, l]) + np.sum(mask[t, :] * lnPi[l, C[t, :] - 1, prev_idx, np.arange(K)])
        
        # normalise
        scaling_factor[t, :] = logsumexp(lnR_[t, :])
        lnR_[t, :] = lnR_[t, :] - scaling_factor[t, :]
            
    return lnR_

def _parallel_forward_pass(C, lnA, lnPi, initProbs, doc_start, before_doc_idx=-1, skip=True):
    '''
    Perform the forward pass of the Forward-Backward algorithm (for multiple documents in parallel).
    '''
    # split into documents
    docs = np.split(C, np.where(doc_start == 1)[0][1:], axis=0)
    # run forward pass for each doc concurrently
    res = Parallel(n_jobs=-1)(delayed(_doc_forward_pass)(doc, lnA, lnPi, initProbs, before_doc_idx, skip) for doc in docs)
    # reformat results
    lnR_ = np.concatenate(res, axis=0)
    
    return lnR_
    
        
def _backward_pass(C, lnA, lnPi, doc_start, before_doc_idx=-1, skip=True):
    '''
    Perform the backward pass of the Forward-Backward algorithm (for each document separately).
    '''
    # infer number of tokens, labels, and annotators
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnPi.shape[-1]
    
    # initialise variables
    lnLambda = np.zeros((T, L)) 
    
    mask = np.ones_like(C)
    
    if skip:
        mask = (C != 0)
    
    # iterate through all tokens, starting at the end going backwards
    for t in xrange(T - 1, -1, -1):
        # check whether the next token was a document start
        if t == T - 1 or doc_start[t + 1]:
            lnLambda[t, :] = 0
            continue
        
        # iterate through all possible labels
        for l in xrange(L):              
            
            prev_idx = C[t, :] - 1
            prev_idx[prev_idx == -1] = before_doc_idx              
            
                        
            lnLambda[t, l] = logsumexp(lnA[l, :] + lnLambda[t + 1, :] + np.sum(mask[t + 1, :] * lnPi[:, C[t + 1, :] - 1, prev_idx, np.arange(K)], axis=1)) #- scaling_factor[t + 1, :] 
        
        if(np.any(np.isnan(lnLambda[t, :]))):    
            print 'backward pass: nan value encountered: '
            print lnLambda[t, :]
  
    return lnLambda

def _parallel_backward_pass(C, lnA, lnPi, doc_start, before_doc_idx=-1, skip=True):
    '''
    Perform the backward pass of the Forward-Backward algorithm (for multiple documents in parallel).
    '''
    # split into documents
    docs = np.split(C, np.where(doc_start == 1)[0][1:], axis=0)
    #docs = np.split(C, np.where(doc_start == 1)[0][1:], axis=0)
    # run forward pass for each doc concurrently
    res = Parallel(n_jobs=-1)(delayed(_doc_backward_pass)(doc, lnA, lnPi, before_doc_idx, skip) for doc in docs)
    # reformat results
    lnLambda = np.concatenate(res, axis=0)
    
    return lnLambda

def _doc_backward_pass(C, lnA, lnPi, before_doc_idx=-1, skip=True):
    '''
    Perform the backward pass of the Forward-Backward algorithm (for a single document).
    '''
    # infer number of tokens, labels, and annotators
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnPi.shape[-1]
    
    # initialise variables
    lnLambda = np.zeros((T, L)) 
    
    mask = np.ones_like(C)
    
    if skip:
        mask = (C != 0)
        
    lnLambda[T - 1, :] = 0
    
    # iterate through all tokens, starting at the end going backwards
    for t in xrange(T - 2, -1, -1): 
        
        # iterate through all possible labels
        for l in xrange(L):              
            
            prev_idx = C[t, :] - 1
            prev_idx[prev_idx == -1] = before_doc_idx              
             
            lnLambda[t, l] = logsumexp(lnA[l, :] + lnLambda[t + 1, :] + np.sum(mask[t + 1, :] * lnPi[:, C[t + 1, :] - 1, prev_idx, np.arange(K)], axis=1))# - scaling_factor[t + 1, :] 
        
        if(np.any(np.isnan(lnLambda[t, :]))):    
            print 'backward pass: nan value encountered: '
            print lnLambda[t, :]
  
    return lnLambda
