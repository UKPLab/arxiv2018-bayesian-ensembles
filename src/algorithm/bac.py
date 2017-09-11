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
    T = None # number of tokens
    
    nscores = None
    
    nu0 = None
    
    lnA = None # transition matrix
    lnPi = None # worker confusion matrices
    
    q_t = None
    q_t_old = None
    
    iter = 0 # iteration
    
    max_iter = None
    eps = None
    

    def __init__(self, L=3, K=5, max_iter=100, eps=1e-5, inside_labels=[0]):
        '''
        Constructor
        '''
        # initialise variables
        self.L = L
        self.nscores = L
        self.K = K
        
        self.alpha0 = np.ones((self.L, self.nscores, self.nscores+1, self.K)) + \
                                    1.0 * np.eye(self.L)[:,:,None,None] # dims: previous_anno c[t-1], current_annoc[t], true_label[t], annoator k
    
        self.nu0 = np.ones((L+1, L)) * 1.0
        for inside_label in inside_labels:
            self.alpha0[:, inside_label, [1,-1], :] = 0.1 # don't set this too close to zero because there are some cases where it was possible for annotators to provide invalid sequences.
            self.nu0[[1,-1], inside_label] = np.nextafter(0,1)
        self.nu0[1, 1] += 1.0
        
        # initialise transition and confusion matrices
        self.lnA = np.log(self.nu0 / np.sum(self.nu0, 1)[:, None])
        
        self.lnPi = np.log(self.alpha0 / np.sum(self.alpha0, 1)[:,None,:,:])
        
        self.before_doc_idx = -1 # identifies which true class value is assumed for the label before the start of a document
        
        self.max_iter = max_iter
        self.eps = eps
        
        self.verbose = False # can change this if you want progress updates to be printed
        
            
    def run(self, C, doc_start):
        
        # validate input data
        assert C.shape[0] == doc_start.shape[0]
        
        #
        C = C.astype(int) + 1
        
        doc_start = doc_start.astype(bool)
        
        self.iter = 0
        self.q_t_old = np.zeros((C.shape[0], self.L))
        self.q_t = np.ones((C.shape[0], self.L))
        
        while not self.converged():
            
            if self.verbose:
                print "BAC iteration %i in progress" % self.iter
            
            self.iter += 1
            
            # calculate alphas and betas using forward-backward algorithm 
            r_ = forward_pass(C, self.lnA[0:self.L,:], self.lnPi, self.lnA[-1,:], doc_start, 
                              self.before_doc_idx)
            lambd = backward_pass(C, self.lnA[0:self.L,:], self.lnPi, doc_start)
            
            # update q_t
            self.q_t_old = self.q_t
            self.q_t = expec_t(r_, lambd)
            
            # update q_t_joint
            self.q_t_joint = expec_joint_t(r_, lambd, self.lnA, self.lnPi, C, doc_start, self.before_doc_idx)
            
            # update E_lnA
            self.lnA, nu = calc_q_A(self.q_t_joint, self.nu0)
            
            alpha = post_alpha(self.q_t, C, self.alpha0, doc_start, self.before_doc_idx)

            # update E_lnpi
            self.lnPi = calc_q_pi(alpha)

        return self.most_probable_sequence(C, doc_start)
    
    def most_probable_sequence(self, C, doc_start):
        # Use Viterbi decoding to ensure we make a valid prediction. There
        # are some cases where most probable sequence does not match argmax(self.q_t,1). E.g.:
        # [[0.41, 0.4, 0.19], [[0.41, 0.42, 0.17]].
        # Taking the most probable labels results in an illegal sequence of [0, 1]. 
        # Using most probable sequence would result in [0, 0].
        
        lnV = np.zeros((C.shape[0], self.L))
        prev = np.zeros((C.shape[0], self.L)) # most likely previous states
                
        for t in range(0, C.shape[0]):
            for l in range(self.L):
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
        
        for t in range(C.shape[0]-1, -1, -1):
            if t+1 == C.shape[0] or doc_start[t+1]:
                seq[t] = np.argmax(lnV[t, :])
            else:
                seq[t] = prev[t+1, seq[t+1]]
            pseq[t, :] = np.exp(lnV[t, :] - logsumexp(lnV[t, :]))
        
        return pseq, seq

    def converged(self):
        if self.verbose:
            print "Difference in values at iteration %i: %.5f" % (self.iter, np.max(np.abs(self.q_t_old - self.q_t)))
        return ((self.iter >= self.max_iter) or np.max(np.abs(self.q_t_old - self.q_t)) < self.eps)

def calc_q_A(E_t, nu0):
    nu = nu0 + np.sum(E_t,0)
    q_A = psi(nu) - psi(np.sum(nu, -1))[:, None]
    
    # for last row, use all data points, not just doc starts
    #q_A[-1, :] = psi(np.sum(nu, 0)) - psi(np.sum(nu))
    
    if np.any(np.isnan(q_A)):
        print 'calc_q_A: nan value encountered!'
    
    return q_A, nu

def calc_q_pi(alpha):
    alpha_sum = np.sum(alpha, 1)
    dims = alpha.shape
    q_pi = np.zeros(dims)
    for s in range(dims[1]): 
        q_pi[:, s, :, :] = psi(alpha[:, s, :, :]) - psi(alpha_sum)
        #q_pi[:, s, -1, :] = psi(np.sum(alpha[:, s, :, :], 1)) - psi(np.sum(alpha_sum, 1))
    
    return q_pi
    
def post_alpha(E_t, C, alpha0, doc_start, before_doc_idx=-1):  # Posterior Hyperparameters
    
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

def expec_t(lnR_, lnLambda):
    r_lambd = np.exp(lnR_ + lnLambda)
    return r_lambd/np.sum(r_lambd,1)[:, None]
    
def expec_joint_t(lnR_, lnLambda, lnA, lnPi, C, doc_start, before_doc_idx=-1):
      
    # initialise variables
    T = lnR_.shape[0]
    L = lnA.shape[-1]
    K = lnPi.shape[-1]
    
    lnS = np.repeat(lnLambda[:,None,:], L+1, 1)    
    flags = -np.inf * np.ones_like(lnS)
    
    # iterate though everything, calculate joint expectations
    for t in xrange(T):
        for l_prev in xrange(L):
            for l in xrange(L):
                
                if doc_start[t]:
                    flags[t, before_doc_idx, l] = 0
                    if l_prev==0: # only add these values once
                        lnS[t, -1, l] += lnA[-1, l] + np.sum((C[t,:]!=0) * lnPi[l, C[t,:]-1, before_doc_idx,
                                                                             np.arange(K)])
                else:
                    flags[t, l_prev, l] = 0
                    lnS[t, l_prev, l] += lnR_[t-1, l_prev] + lnA[l_prev, l]
                    lnS[t, l_prev, l] += np.sum((C[t,:]!=0) * lnPi[l, C[t,:]-1, C[t-1,:]-1, np.arange(K)])   

                if np.any(np.isnan(np.exp(lnS[t,l_prev,:]))):
                    print 'expec_joint_t: nan value encountered for t=%i, l_prev=%i' % (t,l_prev)
        
        #normalise and return  
        lnS[t,:,:] = lnS[t,:,:] + flags[t,:,:] 
        if np.any(np.isnan(np.exp(lnS[t,:,:]))):
            print 'expec_joint_t: nan value encountered (1) '
            print logsumexp(lnS[t,:,:])
            print lnS[t,:,:]
        lnS[t,:,:] = lnS[t,:,:] - logsumexp(lnS[t,:,:]) 
        
        if np.any(np.isnan(np.exp(lnS[t,:,:]))):
            print 'expec_joint_t: nan value encountered' 
            print logsumexp(lnS[t,:,:])
            print lnS[t,:,:]
            
    #print lnS
    
    return np.exp(lnS)

def forward_pass(C, lnA, lnPi, initProbs, doc_start, skip=True, before_doc_idx=-1):
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnPi.shape[-1]
    lnR_ = np.zeros((T, L)) 
    
    for t in xrange(T):
        for l in xrange(L):
            
            if doc_start[t]:
                # compute values for document start
                if skip:
                    mask = C[t,:] != 0
                else:
                    mask = 1                
                lnR_[t,l] += initProbs[l] + np.sum(mask * lnPi[l, C[t,:]-1, before_doc_idx, np.arange(K)])
            else:
                if skip:
                    mask = C[t,:] != 0
                else:
                    mask = 1
                lnR_[t,l] = logsumexp(lnR_[t-1,:] + lnA[:,l]) + np.sum(mask * lnPi[l, C[t,:]-1, C[t-1,:]-1, np.arange(K)])                
        
        # normalise
        lnR_[t,:] = lnR_[t,:] - logsumexp(lnR_[t,:])
            
    return lnR_        
        
def backward_pass(C, lnA, lnPi, doc_start, skip=True):
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnPi.shape[-1]
    lnLambda = np.zeros((T,L)) 
    
    for t in xrange(T-1,-1,-1):
        if t==T-1 or doc_start[t+1]:
            lnLambda[t,:] = 0
            continue
        
        for l in xrange(L):              

            lambdaL_next_sum = np.zeros((L,1))
            
            for l_next in xrange(L):
                if skip:
                    mask = C[t+1,:] != 0
                else:
                    mask = 1
                lambdaL_next = lnLambda[t+1,l_next] + np.sum(mask * lnPi[l_next, C[t+1,:]-1, C[t,:], 
                                                                         np.arange(K)])
            
                lambdaL_next += lnA[l,l_next]                        
                lambdaL_next_sum[l_next] = lambdaL_next
            
            lnLambda[t,l] = logsumexp(lambdaL_next_sum)
        
        # normalise
        unnormed_lnLambda = lnLambda[t, :]
        lnLambda[t,:] = lnLambda[t,:] - logsumexp(lnLambda[t,:])   
        
        if(np.any(np.isnan(lnLambda[t,:]))):    
            print 'backward pass: nan value encountered: '
            print unnormed_lnLambda
  
    return lnLambda                
