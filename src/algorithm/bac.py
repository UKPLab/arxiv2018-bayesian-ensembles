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
    

    def __init__(self, L=3, K=5, max_iter=100, eps=1e-5):
        '''
        Constructor
        '''
        # initialise variables
        self.L = L
        self.nscores = L
        self.K = K
        
        self.alpha0 = np.ones((self.L, self.nscores, self.nscores+1, self.K)) + np.eye(self.L)[:,:,None,None] # dims: previous_anno c[t-1], current_annoc[t], true_label[t], annoator k
        
        self.nu0 = np.ones((L+1, L))
        self.nu0[[1,3],0] = np.nextafter(0,1)
        
        # initialise transition and confusion matrices
        self.lnA = np.log(np.ones((L+1, L))/L)
        self.lnA[[1,3],0] = np.log(np.nextafter(0,1))
        #self.lnA[1,0] = np.nextafter(0,1)
        
        self.lnPi = np.log(self.alpha0 / np.sum(self.alpha0, 2)[:,:,None,:])
        self.lnPi[:,0,[3,1],:] = np.log(np.nextafter(0,1))
        
        
        self.max_iter = max_iter
        self.eps = eps
        
            
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
            
            self.iter += 1
            
            # calculate alphas and betas using forward-backward algorithm 
            r_ = forward_pass(C, self.lnA[0:self.L,:], self.lnPi, self.lnA[-1,:], doc_start)
            lambd = backward_pass(C, self.lnA[0:self.L,:], self.lnPi, doc_start)
            
            # update q_t
            self.q_t_old = self.q_t
            self.q_t = expec_t(r_, lambd)
            
            # update q_t_joint
            self.q_t_joint = expec_joint_t(r_, lambd, self.lnA, self.lnPi, C, doc_start)
            
            # update E_lnA
            self.lnA, nu = calc_q_A(self.q_t_joint, self.nu0)
            
            alpha = post_alpha(self.q_t, C, self.alpha0, doc_start)
            
            # update E_lnpi
            self.lnPi = calc_q_pi(alpha)
            
        return self.q_t #, self.most_probable_sequence(C, doc_start)
    
            
    def converged(self):
        return ((self.iter >= self.max_iter) or np.max(np.abs(self.q_t_old - self.q_t)) < self.eps)
    
    
def calc_q_A(E_t, nu0):
    nu = nu0 + np.reshape(np.sum(E_t,0), nu0.shape)
    q_A = (psi(nu).T - psi(np.sum(nu, -1))).T
    
    if np.any(np.isnan(q_A)):
        print 'calc_q_A: nan value encountered!'
    
    return q_A, nu

def calc_q_pi(alpha):
    alpha_sum = np.sum(alpha, 1)
    dims = alpha.shape
    q_pi = np.zeros(dims)
    for s in range(dims[1]): 
        q_pi[:, s, :, :] = psi(alpha[:, s, :, :]) - psi(alpha_sum)
    
    return q_pi
    
def post_alpha(E_t, C, alpha0, doc_start):  # Posterior Hyperparameters
    
    dims = alpha0.shape
    
    alpha = alpha0.copy()
    
    for j in range(dims[0]): 
        Tj = E_t[:, j] 
        for l in range(dims[1]): 
            
            counts = ((C==l+1) * doc_start).T.dot(Tj).reshape(-1)    
            alpha[j, l, -1, :] = alpha0[j, l, -1, :] + counts
            
            for m in range(dims[1]): 
                
                #counts = ((C==l+1)[1:,:] * np.invert(doc_start[1:]) * (C==m+1)[:-1, :]).T.dot(Tj[1:]).reshape(-1)
                counts = ((C==l+1)[1:,:] * (1-doc_start[1:]) * (C==m+1)[:-1, :]).T.dot(Tj[1:]).reshape(-1)
                
                alpha[j, l, m, :] = alpha[j, l, m, :] + counts
    
    return alpha

def expec_t(lnR_, lnLambda):
    r_ = np.exp(lnR_)
    lambd = np.exp(lnLambda)
    return (r_*lambd)/np.sum(r_*lambd,1)[:, None]

    
def expec_joint_t(lnR_, lnLambda, lnA, lnPi, C, doc_start):
      
    # initialise variables
    T = lnR_.shape[0]
    L = lnA.shape[-1]
    K = lnPi.shape[-1]
    
    lnS = np.repeat(lnLambda[:,None,:], L+1, 1)
    flags = -np.inf * np.ones_like(lnS)
    
    # iterate though everything, calculate joint expectations
    for t in xrange(T):
        for l in xrange(L):
            for l_next in xrange(L):
                
                if doc_start[t]:
                    flags[t, -1, l_next] = 0
                    lnS[t, -1, l_next] += lnA[-1, l_next]
                else:
                    flags[t, l, l_next] = 0
                    lnS[t, l, l_next] += lnR_[t-1, l] + lnA[l, l_next]
                    
                for k in xrange(K):
                    
                    if doc_start[t]:
                        
                        
                        if C[t,k]==0: # ignore unannotated tokens
                            continue
                        lnS[t, -1, l_next] += lnPi[l, C[t,k]-1, -1, k]
                    else:
                        
                        
                        if C[t,k]==0: # ignore unannotated tokens
                            continue
                        lnS[t, l, l_next] += lnPi[l, C[t,k]-1, C[t-1,k]-1, k]
            
        
        #normalise and return  
        lnS[t,:,:] = lnS[t,:,:] + flags[t,:,:] 
        lnS[t,:,:] = lnS[t,:,:] - logsumexp(lnS[t,:,:]) 
        
    #print lnS
    
    return np.exp(lnS)

def forward_pass(C, lnA, lnPi, initProbs, doc_start, skip=True):
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnPi.shape[-1]
    lnR_ = np.zeros((T, L)) 
    
    for t in xrange(T):
        for l in xrange(L):
            
            if doc_start[t]:
                # compute values for document start
                
                lnR_[t,l] += initProbs[l] 
                
                for k in xrange(K):
                    # skip unannotated tokens
                    if skip and C[t,k] == 0:
                        continue
                    lnR_[t,l] += lnPi[l, C[t,k]-1, -1, k]
            else:
                for l_prev in xrange(L):
                    lnR_[t,l] += np.exp(lnR_[t-1,l_prev] + lnA[l_prev,l])
                
                lnR_[t,l] = np.log(lnR_[t,l])
                
                for k in xrange(K):
                    # skip unannotated tokens
                    if skip and C[t,k] == 0:
                        continue
                    
                    lnR_[t,l] += lnPi[l, C[t,k]-1, C[t-1,k]-1, k]
        
        # normalise
        lnR_[t,:] = lnR_[t,:] - logsumexp(lnR_[t,:])
            
    return lnR_        

        
def backward_pass(C, lnA, lnPi, doc_start, skip=True):
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnPi.shape[-1]
    lnLambda = np.zeros((T,L)) 
    
    for t in xrange(T-2,-1,-1):
        for l in xrange(L):              
                 
            lambdaL_next_sum = 0
                
            for l_next in xrange(L):
                
                if t==T-1 or doc_start[t+1]:
                    lambdaL_next = 0
                else: 
                    lambdaL_next = lnLambda[t+1,l_next]
                
                lambdaL_next += lnA[l,l_next]
                
                for k in xrange(K):
                    
                    # skip unannotated tokens
                    if skip and C[t,k] == 0:
                        continue
            
                    lambdaL_next += lnPi[l_next, C[t+1,k]-1, C[t,k]-1, k]
                    
                lambdaL_next_sum += np.exp(lambdaL_next)
            
            lnLambda[t,l] += np.log(lambdaL_next_sum)
            
        
        # normalise  
        lnLambda[t,:] = lnLambda[t,:] - logsumexp(lnLambda[t,:])                
  
    return lnLambda                