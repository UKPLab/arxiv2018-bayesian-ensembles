'''
Created on Dec 20, 2016

@author: Melvin Laux
'''
import numpy as np

def expec_t(r_, lambd):
    return (r_*lambd)/np.sum(r_*lambd,1)[:, None]
    
def expec_joint_t(r_, lambd, lnA, lnPi, C, doc_start):
    
    # initialise variables
    T = r_.shape[0]
    L = lnA.shape[0]
    K = lnA.shape[-1]
    s = np.zeros((T, L+1, L))
    
    # iterate though everything, calculate joint expectations
    for t in xrange(T):
        for l in xrange(L):
            for l_next in xrange(L):
                for k in xrange(K):
                    
                    if C[t,k]==0: # ignore unannotated tokens
                        continue
                    
                    if doc_start[t]:
                        s[t,-1,l_next] += r_[t,l] * lambd[t,l_next] * lnA[l,l_next] * lnPi[l,int(C[t,k])-1,-1, k]
                    else:
                        s[t,l,l_next] += r_[t,l] * lambd[t,l_next] * lnA[l,l_next] * lnPi[l,int(C[t,k])-1,int(C[t-1,k])-1, k]
    
    #normalise and return            
    return s / np.sum(np.sum(s, 2), 1)[:,None, None]
    
                

if __name__ == '__main__':
    pass