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
    lnR_ = np.zeros((T,L)) 
    
    # compute values for t=1
    
    #lnR_[0,:] = lnR_[0,:]/sum(lnR_[0,:])
    
    for t in xrange(T):
        for l in xrange(L):
            
            if doc_start[t]:
                # compute values for document start
                for k in xrange(K):
                    # skip unannotated tokens
                    if skip and C[t,k] == 0:
                        continue
                    lnR_[t,l] += initProbs[l] + lnPi[l, int(C[t,k])-1, -1, k]
            else:
                for l_prev in xrange(L):
                    lnR_[t,l] += np.exp(lnR_[t-1,l_prev] + lnA[l_prev,l])
                
                lnR_[t,l] = np.log(lnR_[t,l])
                
                for k in xrange(K):
                    # skip unannotated tokens
                    if skip and C[t,k] == 0:
                        continue
                    
                    lnR_[t,l] += lnPi[l,int(C[t,k])-1,int(C[t-1,k])-1, k]
        
        # normalise
        r_ = np.exp(lnR_[t,:])
        normR_ = r_/np.sum(r_)    
        lnR_[t,:] = np.log(normR_) #lnR_[t,:]/lnR_[t,:]
            
    return lnR_        

        
def backward_pass(C, lnA, lnPi, doc_start, skip=True):
    T = C.shape[0]
    L = lnA.shape[0]
    K = lnPi.shape[-1]
    lnLambda = np.zeros((T,L)) 
    
    # set values for t=T
    #lnLambda[T-1,:] = np.ones((L,))
    
    for t in xrange(T-1,-1,-1):
        
        # set values for t=T
        #if doc_start[t+1]:
        #    lnLambda[t,:] = np.ones((L,))
        #    continue
        
        for l in xrange(L):  
            
                 
            lambdaL_next_sum = 0
                
            for l_next in xrange(L):
                #lnLambda[t,l] += lnA[l,l_next]
                if t==T-1 or doc_start[t+1]:
                    #lnLambda[t,l] = 1
                    lambdaL_next = 0
                else: 
                    lambdaL_next = lnLambda[t+1,l_next]
                
                lambdaL_next += lnA[l,l_next]
                
                for k in xrange(K):
                    
                    # skip unannotated tokens
                    if skip and C[t,k] == 0:
                        continue
                    
                    #lnLambda[t,l] += lnPi[l_next, int(C[t,k])-1, int(C[t-1,k])-1, k]
                    lambdaL_next += lnPi[l_next, int(C[t,k])-1, int(C[t-1,k])-1, k]
                    
                lambdaL_next_sum += np.exp(lambdaL_next)
            
            #print lnLambda[t,l]
            lnLambda[t,l] += np.log(lambdaL_next_sum)
            
        
        # normalise  
        #lnLambda = np.exp(lnLambda)      
        #lnLambda[t,:] = lnLambda[t,:]/sum(lnLambda[t,:])                
  
    return lnLambda
    
    
    
if __name__ == '__main__':
    pass