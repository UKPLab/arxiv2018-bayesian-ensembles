'''
Created on Dec 20, 2016

@author: Melvin Laux
'''
import numpy as np

def expec_t(lnR_, lnLambda):
    r_ = np.exp(lnR_)
    lambd = np.exp(lnLambda)
    return (r_*lambd)/np.sum(r_*lambd,1)[:, None]

    
def expec_joint_t(lnR_, lnLambda, lnA, lnPi, C, doc_start):
    
    # initialise variables
    T = lnR_.shape[0]
    L = lnA.shape[-1]
    K = lnPi.shape[-1]
    
    lnS = np.zeros((T, L+1, L))
    
    # iterate though everything, calculate joint expectations
    for t in xrange(T):
        for l in xrange(L):
            for l_next in xrange(L):
                for k in xrange(K):
                    
                    if C[t,k]==0: # ignore unannotated tokens
                        continue
                    
                    if doc_start[t]:
                        lnS[t,-1,l_next] += lnR_[t,l] + lnLambda[t,l_next] + lnA[l,l_next] + lnPi[l,int(C[t,k])-1,-1, k]
                    else:
                        lnS[t,l,l_next] += lnR_[t,l] + lnLambda[t,l_next] + lnA[l,l_next] + lnPi[l,int(C[t,k])-1,int(C[t-1,k])-1, k]
    
    #normalise and return  
    s = np.exp(lnS)          
    return s / np.sum(np.sum(s, 2), 1)[:,None, None]
    
                

if __name__ == '__main__':
    pass