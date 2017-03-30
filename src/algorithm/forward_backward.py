'''
Created on Jan 15, 2017

@author: Melvin Laux
'''

from __future__ import division
import numpy as np


def forward_pass(C, lnA, lnPi, initProbs, doc_start,skip=True):
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnPi.shape[-1]
    r_ = np.zeros((T,L)) 
    
    # compute values for t=1
    
    #r_[0,:] = r_[0,:]/sum(r_[0,:])
    
    for t in xrange(T):
        for l in xrange(L):
            
            if doc_start[t]:
                # compute values for document start
                for k in xrange(K):
                    # skip unannotated tokens
                    if skip and C[t,k] == 0:
                        continue
                    r_[t,l] += initProbs[l] * np.exp(lnPi[l, int(C[t,k])-1, -1, k])
            else:
                for l_prev in xrange(L):
                    r_[t,l] += r_[t-1,l_prev] * np.exp(lnA[l_prev,l])
                
                for k in xrange(K):
                    # skip unannotated tokens
                    if skip and C[t,k] == 0:
                        continue
                    
                    r_[t,l] += np.exp(lnPi[l,int(C[t,k])-1,int(C[t-1,k])-1, k])
            
        r_[t,:] = r_[t,:]/sum(r_[t,:])
            
    return r_        

        
def backward_pass(C, lnA, lnPi, doc_start, skip=True):
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnPi.shape[-1]
    lambd = np.zeros((T,L)) 
    
    # set values for t=T
    #lambd[T-1,:] = np.ones((L,))
    
    for t in xrange(T-1,-1,-1):
        
        # set values for t=T
        #if doc_start[t+1]:
        #    lambd[t,:] = np.ones((L,))
        #    continue
        
        for l in xrange(L):            
            for l_next in xrange(L):
                for k in xrange(K):
                    
                    # skip unannotated tokens
                    if skip and C[t,k] == 0:
                        continue
                    
                    if t==T-1 or doc_start[t+1]:
                        #lambd[t,l] += lambd[t+1,l_next] * np.exp(lnA[l,l_next]) * np.exp(lnPi[l_next, int(C[t,k])-1, -1, k])
                        lambd[t,l] += np.exp(lnA[l,l_next]) * np.exp(lnPi[l_next, int(C[t,k])-1, -1, k])
                    else:
                        lambd[t,l] += lambd[t+1,l_next] * np.exp(lnA[l,l_next]) * np.exp(lnPi[l_next, int(C[t,k])-1, int(C[t-1,k])-1, k])
                
        lambd[t,:] = lambd[t,:]/sum(lambd[t,:])                
  
    return lambd
    
    
    
if __name__ == '__main__':
    pass