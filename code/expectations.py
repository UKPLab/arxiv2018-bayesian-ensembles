'''
Created on Dec 20, 2016

@author: Melvin Laux
'''
import numpy as np

def expec_s(alpha, beta):
    return (alpha*beta)/np.sum(alpha*beta,0)
    
def expec_joint_s(alpha, beta, A, C, y):
    
    # initialise variables
    (T, num_states) = alpha.shape
    E_joint_s = np.zeros((T, num_states, num_states))
    
    # iterate though everything, calculate joint expectations
    for t in xrange(T):
        for j in xrange(num_states):
            for j_prime in xrange(num_states):
                E_joint_s[t,j,j_prime] = alpha[t,j]*A[j,j_prime]*C[j,int(y[t])]*beta[t,j_prime]
    
    #normalise and return            
    return E_joint_s / np.sum(np.sum(E_joint_s, 2), 1)[:,None, None]
    
                

if __name__ == '__main__':
    pass