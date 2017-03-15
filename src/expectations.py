'''
Created on Dec 20, 2016

@author: Melvin Laux
'''
import numpy as np

def expec_t(alpha, beta):
    return (alpha*beta)/np.sum(alpha*beta,1)[:, None]
    
def expec_joint_t(alpha, beta, lnA, lnPi, C, doc_start):
    
    # initialise variables
    T = alpha.shape[0]
    L = lnA.shape[0]
    K = lnA.shape[-1]
    E_joint_t = np.zeros((T, L+1, L))
    
    # iterate though everything, calculate joint expectations
    for t in xrange(T):
        for l in xrange(L):
            for l_next in xrange(L):
                for k in xrange(K):
                    
                    if C[t,k]==0: # ignore unlabelled tokens
                        continue
                    
                    if doc_start[t]:
                        E_joint_t[t,-1,l_next] += alpha[t,l]*lnA[l,l_next]*lnPi[l,int(C[t,k]),-1, k]*beta[t,l_next]
                    else:
                        E_joint_t[t,l,l_next] += alpha[t,l]*lnA[l,l_next]*lnPi[l,int(C[t,k]),int(C[t-1,k]), k]*beta[t,l_next]
    
    #normalise and return            
    return E_joint_t / np.sum(np.sum(E_joint_t, 2), 1)[:,None, None]
    
                

if __name__ == '__main__':
    pass