'''
Created on Jan 15, 2017

@author: Melvin Laux
'''

from __future__ import division
import numpy as np


def forward_pass(C, lnA, lnPi, initProbs, doc_start):
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnA.shape[-1]
    r_ = np.zeros((T,L)) 
    
    # compute values for t=1
    
    #r_[0,:] = r_[0,:]/sum(r_[0,:])
    
    for t in xrange(T):
        for l in xrange(L):
            
            
            if doc_start[t]:
                # compute values for document start
                for k in xrange(K):
                    r_[t,l] = initProbs[l] * lnPi[l, C[t,k], -1, k]    
            else:
                for l_prev in xrange(L):
                    r_[t,l] = r_[t,l] + r_[t-1,l_prev]*lnA[l_prev,l]
                
                #r_[t,l] = r_[t,l] * lnPi[l,C[t]]
                for k in xrange(K):
                    r_[t,l] = r_[t,l] * lnPi[l,C[t,k],C[t-1,k], k]
            
        r_[t,:] = r_[t,:]/sum(r_[t,:])
            
    return r_        

        
def backward_pass(C, lnA, lnPi, doc_start):
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnA.shape[-1]
    lambd = np.zeros((T,L)) 
    
    # compute values for t=T
    for l in xrange(L):
        lambd[T-1,l] = 1
    
    for t in xrange(T-2,-1,-1):        
        for l in xrange(L):            
            for l_next in xrange(L):
                #lambd[t,l] = lambd[t,l] + lambd[t+1,l_next]*lnA[l,l_next]*lnPi[l_next,C[t]]
                
                for k in xrange(K):
                    if doc_start[t]:
                        lambd[t,l] = lambd[t,l] + lambd[t+1,l_next]*lnA[l,l_next]*lnPi[l_next, C[t,k], -1, k]
                    else:
                        lambd[t,l] = lambd[t,l] + lambd[t+1,l_next]*lnA[l,l_next]*lnPi[l_next, C[t,k], C[t-1,k], k]
                
        lambd[t,:] = lambd[t,:]/sum(lambd[t,:])                
  
    return lambd
    
    
    
if __name__ == '__main__':
    pass