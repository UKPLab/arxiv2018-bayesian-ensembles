'''
Created on Jan 28, 2017

@author: Melvin Laux
'''

import numpy as np
from scipy.misc import logsumexp
from scipy.special import psi

class BAC(object):
    '''
    classdocs
    '''
    
    K = None # number of annotators
    L = None # number of class labels
    
    nscores = None # number of possible values a token can have, usually this is L + 1 (add one to account for unannotated tokens)
    
    nu0 = None # ground truth priors
    
    lnA = None # transition matrix
    lnPi = None # worker confusion matrices
    
    q_t = None # current true label estimates
    q_t_old = None # previous true labels estimates
    
    iter = 0 # current iteration
    
    max_iter = None # maximum number of iterations
    eps = None # maximum difference of estimate differences in convergence chack
    

    def __init__(self, L=3, K=5, max_iter=100, eps=1e-5, inside_labels=[0], exclusions=None, alpha0=None, nu0=None):
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
            self.alpha0 = np.ones((self.L, self.nscores, self.nscores+1, self.K)) +  1.0 * np.eye(self.L)[:,:,None,None] 
        else:
            self.alpha0 = alpha0
    
        if nu0 is None:
            self.nu0 = np.ones((L+1, L))
        else:
            self.nu0 = nu0
        
        # set priors for invalid transitions (to low values)
        for inside_label in inside_labels:
            self.alpha0[:, inside_label, [1,-1], :] = 0.1 # don't set this too close to zero because there are some cases where it was possible for annotators to provide invalid sequences.
            self.nu0[[1,-1], inside_label] = np.nextafter(0,1)
        #self.nu0[1, 1] += 1.0
        
        if exclusions is not None:
            for label,excluded in dict(exclusions).iteritems():
                self.alpha0[:,excluded,label,:] = np.nextafter(0,1)
                self.nu0[label,excluded] = np.nextafter(0,1)
            
        
        # initialise transition and confusion matrices
        self.lnA = np.log(self.nu0 / np.sum(self.nu0, 1)[:, None])
        self.lnPi = np.log(self.alpha0 / np.sum(self.alpha0, 1)[:,None,:,:])
        
        self.before_doc_idx = -1 # identifies which true class value is assumed for the label before the start of a document
        
        self.max_iter = max_iter # maximum number of iterations
        self.eps = eps # threshold for convergence 
        
        self.verbose = False # can change this if you want progress updates to be printed
        
            
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
        
        # main inference loop
        while not self._converged():
            
            # print status if desired
            if self.verbose:
                print "BAC iteration %i in progress" % self.iter
            
            # calculate alphas and betas using forward-backward algorithm 
            r_ = _forward_pass(C, self.lnA[0:self.L,:], self.lnPi, self.lnA[-1,:], doc_start, 
                              self.before_doc_idx)
            lambd = _backward_pass(C, self.lnA[0:self.L,:], self.lnPi, doc_start)
            
            # update q_t
            self.q_t_old = self.q_t
            self.q_t = _expec_t(r_, lambd)
            
            # update q_t_joint
            self.q_t_joint = _expec_joint_t(r_, lambd, self.lnA, self.lnPi, C, doc_start, self.before_doc_idx)
            
            # update E_lnA
            self.lnA, _ = _calc_q_A(self.q_t_joint, self.nu0)
            
            alpha = _post_alpha(self.q_t, C, self.alpha0, doc_start, self.before_doc_idx)

            # update E_lnpi
            self.lnPi = _calc_q_pi(alpha) 
            
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
        prev = np.zeros((C.shape[0], self.L)) # most likely previous states
                
        for t in range(0, C.shape[0]):
            for l in xrange(self.L):
                if doc_start[t]:
                    likelihood_current = np.sum((C[0, :]!=0) * self.lnPi[l, C[0, :]-1, self.before_doc_idx, 
                                                                         np.arange(self.K)])
                    lnV[t, l] = self.lnA[self.before_doc_idx, l] + likelihood_current
                else:
                    likelihood_current = np.sum((C[t, :]!=0) * self.lnPi[l, C[t, :]-1, C[t-1, :]-1, np.arange(self.K)]) 
                    p_current = lnV[t-1, :] + self.lnA[:self.L, l] + likelihood_current
                    lnV[t, l] = np.max(p_current)
                    prev[t, l] = np.argmax(p_current, axis=0)
            
        # decode
        seq = np.zeros(C.shape[0], dtype=int)
        pseq = np.zeros((C.shape[0], self.L), dtype=float)
        
        for t in xrange(C.shape[0]-1, -1, -1):
            if t+1 == C.shape[0] or doc_start[t+1]:
                seq[t] = np.argmax(lnV[t, :])
            else:
                seq[t] = prev[t+1, seq[t+1]]
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
    nu = nu0 + np.sum(E_t,0)
    q_A = psi(nu) - psi(np.sum(nu, -1))[:, None]
    
    if np.any(np.isnan(q_A)):
        print '_calc_q_A: nan value encountered!'
    
    return q_A, nu

def _calc_q_pi(alpha):
    '''
    Update the annotator models.
    '''
    alpha_sum = np.sum(alpha, 1)
    dims = alpha.shape
    q_pi = np.zeros(dims)
    for s in range(dims[1]): 
        q_pi[:, s, :, :] = psi(alpha[:, s, :, :]) - psi(alpha_sum)
    
    return q_pi
    
def _post_alpha(E_t, C, alpha0, doc_start, before_doc_idx=-1):  # Posterior Hyperparameters
    '''
    Update alpha.
    '''
    
    dims = alpha0.shape
    
    alpha = alpha0.copy()
    
    for j in range(dims[0]): 
        Tj = E_t[:, j] 
        for l in range(dims[1]): 
            
            counts = ((C==l+1) * doc_start).T.dot(Tj).reshape(-1)    
            alpha[j, l, before_doc_idx, :] = alpha0[j, l, before_doc_idx, :] + counts
            
            for m in range(dims[1]): 
                counts = ((C==l+1)[1:,:] * (1-doc_start[1:]) * (C==m+1)[:-1, :]).T.dot(Tj[1:]).reshape(-1)
                alpha[j, l, m, :] = alpha[j, l, m, :] + counts
    
    return alpha

def _expec_t(lnR_, lnLambda):
    '''
    Calculate label probabilities for each token.
    '''
    r_lambd = np.exp(lnR_ + lnLambda)
    return r_lambd/np.sum(r_lambd,1)[:, None]
    
def _expec_joint_t(lnR_, lnLambda, lnA, lnPi, C, doc_start, before_doc_idx=-1):
    '''
    Calculate joint label probabilities for each pair of tokens.
    '''  
    # initialise variables
    T = lnR_.shape[0]
    L = lnA.shape[-1]
    K = lnPi.shape[-1]
    
    lnS = np.repeat(lnLambda[:,None,:], L+1, 1)    
    lnS_test = lnS
    flags = -np.inf * np.ones_like(lnS)
    flags_test = flags
    
    # iterate though everything, calculate joint expectations
    for t in xrange(T):
        for l_prev in xrange(L):
            
            if doc_start[t]:
                flags_test[t,before_doc_idx,:] = 0
                lnS_test[t,before_doc_idx,:] += np.dot(lnPi[:,C[t,:]-1,before_doc_idx,np.arange(K)],(C[t,:]!=0)) + lnA[-1,:]
            else:
                flags_test[t, l_prev,:] = 0 
                lnS_test[t,l_prev,:] += lnR_[t-1,l_prev] + lnA[l_prev,:] + np.dot(lnPi[:,C[t,:]-1,C[t-1,:]-1,np.arange(K)],(C[t,:]!=0))
                
 #           for l in xrange(L):
                
                # check if current token is at document start
  #              if doc_start[t]:
  #                  flags[t, before_doc_idx, l] = 0
  #                  if l_prev==0: # only add these values once
  #                      lnS[t, -1, l] += lnA[-1, l] + np.sum((C[t,:]!=0) * lnPi[l, C[t,:]-1, before_doc_idx,
 #                                                                            np.arange(K)])
                # TODO: replace with np.dot(lnPi[:,C[t,:]-1,before_doc_idx,np.arange(K)],(C[t,:]!=0)) + lnA[-1,:] + np.repeat(lnLambda[:,None,:], L+1, 1) [t,-1,:]
                
  #              else:
  #                  flags[t, l_prev, l] = 0
  #                  lnS[t, l_prev, l] += lnR_[t-1, l_prev] + lnA[l_prev, l]
   #                 lnS[t, l_prev, l] += np.sum((C[t,:]!=0) * lnPi[l, C[t,:]-1, C[t-1,:]-1, np.arange(K)])   
                    
                # TODO: replace with np.repeat(lnLambda[:,None,:], L+1, 1) [t,l_prev,:] + 

  #              if np.any(np.isnan(np.exp(lnS[t,l_prev,:]))):
  #                  print '_expec_joint_t: nan value encountered for t=%i, l_prev=%i' % (t,l_prev)
        
        #normalise and return  
        lnS[t,:,:] = lnS[t,:,:] + flags[t,:,:] 
        lnS_test[t,:,:] = lnS_test[t,:,:] + flags_test[t,:,:] 
        if np.any(np.isnan(np.exp(lnS[t,:,:]))):
            print '_expec_joint_t: nan value encountered (1) '
            print logsumexp(lnS[t,:,:])
            print lnS[t,:,:]
            
        lnS[t,:,:] = lnS[t,:,:] - logsumexp(lnS[t,:,:]) 
        
        if np.any(np.isnan(np.exp(lnS[t,:,:]))):
            print '_expec_joint_t: nan value encountered' 
            print logsumexp(lnS[t,:,:])
            print lnS[t,:,:]
            
        if np.any(lnS_test[t,:,:] != lnS[t,:,:]) or np.any(flags[t,:,:]!=flags_test[t,:,:]):
            print 'Vectorisation Speedup Error!'
    
    return np.exp(lnS_test)

def _forward_pass(C, lnA, lnPi, initProbs, doc_start, skip=True, before_doc_idx=-1):
    '''
    Perform the forward pass of the Forward-Backward algorithm (for each document separately).
    '''
    T = C.shape[0] # infer number of tokens
    L = lnA.shape[0] # infer number of labels
    K = lnPi.shape[-1] # infer number of annotators
    
    # initialise variables
    lnR_ = np.zeros((T, L)) 
    
    # iterate through all tokens, starting at the beginning going forward
    for t in xrange(T):
        # iterate through all possible labels
        for l in xrange(L):
            
            # check if the current token marks the start of a new document
            if doc_start[t]:
                # check if token needs to be skipped
                if skip:
                    mask = C[t,:] != 0
                else:
                    mask = 1    
                
                # compute likelihood for document start token being the current label           
                lnR_[t,l] += initProbs[l] + np.sum(mask * lnPi[l, C[t,:]-1, before_doc_idx, np.arange(K)])
            else:
                # check if token needs to be skipped
                if skip:
                    mask = C[t,:] != 0
                else:
                    mask = 1
                
                # compute likelihood for current token and label
                lnR_[t,l] = logsumexp(lnR_[t-1,:] + lnA[:,l]) + np.sum(mask * lnPi[l, C[t,:]-1, C[t-1,:]-1, np.arange(K)])                
        
        # normalise
        lnR_[t,:] = lnR_[t,:] - logsumexp(lnR_[t,:])
            
    return lnR_        
        
def _backward_pass(C, lnA, lnPi, doc_start, skip=True):
    '''
    Perform the backward pass of the Forward-Backward algorithm (for each document separately).
    '''
    # infer number of tokens, labels, and annotators
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnPi.shape[-1]
    
    # initialise variables
    lnLambda = np.zeros((T,L)) 
    
    # iterate through all tokens, starting at the end going backwards
    for t in xrange(T-1,-1,-1):
        # check whether the previous token was a document start
        if t==T-1 or doc_start[t+1]:
            lnLambda[t,:] = 0
            continue
        
        # iterate through all possible labels
        for l in xrange(L):              
            
            # initialise variable
            lambdaL_next_sum = np.zeros((L,1))
            
            # iterate through all possible labels (again)
            for l_next in xrange(L):
                # check if the token needs to be skipped
                if skip:
                    mask = C[t+1,:] != 0
                else:
                    mask = 1
                    
                # calculate likelihood of current token and label
                lambdaL_next = lnLambda[t+1,l_next] + np.sum(mask * lnPi[l_next, C[t+1,:]-1, C[t,:]-1, 
                                                                         np.arange(K)])
                lambdaL_next += lnA[l,l_next]                        
                lambdaL_next_sum[l_next] = lambdaL_next
                
                # TODO: why in 3 steps?
            
            lnLambda[t,l] = logsumexp(lambdaL_next_sum)
        
        # normalise
        unnormed_lnLambda = lnLambda[t, :]
        lnLambda[t,:] = lnLambda[t,:] - logsumexp(lnLambda[t,:])   
        
        if(np.any(np.isnan(lnLambda[t,:]))):    
            print 'backward pass: nan value encountered: '
            print unnormed_lnLambda
  
    return lnLambda                
