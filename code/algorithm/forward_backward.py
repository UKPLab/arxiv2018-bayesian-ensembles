'''
Created on Jan 15, 2017

@author: Melvin Laux
'''

import numpy as np

def forward_pass(y, A, C, pi):
    T = y.shape[0]
    k = A.shape[0]
    alphas = np.zeros((T,k)) 
    
    # compute values for t=1
    for s in xrange(k):
        alphas[0,s] = pi[s]*C[s,y[0]]
    
    alphas[0,:] = alphas[0,:]/sum(alphas[0,:])
    
    for t in xrange(1,T):
        
        for s in xrange(k):
            
            for s_prev in xrange(k):
                alphas[t,s] = alphas[t,s] + alphas[t-1,s_prev]*A[s_prev,s]
                
            alphas[t,s] = alphas[t,s] * C[s,y[t]]
            
        alphas[t,:] = alphas[t,:]/sum(alphas[t,:])
            
    return alphas        

        
def backward_pass(y, A, C, pi):
    T = y.shape[0]
    k = A.shape[0]
    betas = np.zeros((T+1,k)) 
    
    # compute values for t=T
    for s in xrange(k):
        betas[T,s] = 1
    
    for t in xrange(T-1,-1,-1):
        
        for s in xrange(k):
            
            for s_next in xrange(k):
                betas[t,s] = betas[t,s] + betas[t+1,s_next]*A[s,s_next]*C[s_next,y[t]]
                
        betas[t,:] = betas[t,:]/sum(betas[t,:])                
  
    return betas
    
    
    
if __name__ == '__main__':
    pass