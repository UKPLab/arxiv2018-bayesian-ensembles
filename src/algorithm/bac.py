'''
Created on Jan 28, 2017

@author: Melvin Laux
'''

import numpy as np
import src.algorithm.forward_backward as fb
from src.expectations import expec_t, expec_joint_t
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
    
    lnA = None
    lnPi = None
    
    iter = 0
    

    def __init__(self, L=3, K=5, T=10):
        '''
        Constructor
        '''
        # initialise variables
        self.T = T
        self.L = L
        self.nscores = L
        self.K = K
        
        self.alpha0 = np.ones((self.L, self.nscores, self.nscores+1, self.K)) + np.eye(self.L)[:,:,np.newaxis,np.newaxis] # dims: previous_anno c[t-1], current_annoc[t], true_label[t], annoator k
        self.nu0 = np.log(np.ones((L+1, L)))
        
        # initialise transition and confusion matrices
        self.lnA = np.log(np.ones((L+1, L))/L)
        
        self.lnPi = np.log(self.alpha0 / np.sum(self.alpha0, 2)[:,:,np.newaxis,:])
        
            
    def run(self, C, doc_start):
        
        # validate input data
        assert C.shape[0] == doc_start.shape[0]
        
        doc_start = doc_start.astype(bool)
        
        while not self.converged():
            
            self.iter += 1
            
            print self.iter
            
            # calculate alphas and betas using forward-backward algorithm 
            r_ = fb.forward_pass(C, self.lnA[0:self.L,:], self.lnPi, self.lnA[-1,:], doc_start)
            lambd = fb.backward_pass(C, self.lnA[0:self.L,:], self.lnPi, doc_start)
            
            # update q_t
            self.q_t = expec_t(r_, lambd)
            
            # update q_t_joint
            self.q_t_joint = expec_joint_t(r_, lambd, self.lnA[0:self.L,:], self.lnPi, C, doc_start)
            
            # update E_lnA
            self.lnA, nu = calc_q_A(self.q_t_joint, self.nu0)
            
            alpha = post_alpha(self.q_t, C, self.alpha0, doc_start)
            
            # update E_lnpi
            self.lnPi = calc_q_pi(alpha)
            
        return self.q_t
        
        
            
    def converged(self):
        return (self.iter >= 10)
    
def calc_q_A(E_t, nu0):
    nu = nu0 + np.reshape(np.sum(E_t,0), nu0.shape)
    q_A = (psi(nu).T - psi(np.sum(nu, -1))).T
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
    
    for j in range(dims[0]): # iterate through previous anno
        Tj = E_t[:, j] 
        for l in range(dims[1]): # iterate through current anno
            
            counts = ((C==l+1) * doc_start).T.dot(Tj).reshape(-1)    
            alpha[j, l, -1, :] = alpha0[j, l, -1, :] + counts
            
            for m in range(-1, dims[1]): # iterate through true anno
                
                counts = ((C==l+1)[1:,:] * np.invert(doc_start[1:]) * (C==m+1)[:-1, :]).T.dot(Tj[1:]).reshape(-1)
                
                alpha[j, l, m, :] = alpha[j, l, m, :] + counts
    
    return alpha
            
            