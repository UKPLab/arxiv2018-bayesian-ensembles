'''
Created on 10 May 2016

@author: simpson
'''

import numpy as np

def alpha(U, C, L):
    '''
    U - units of analysis, i.e. the data points being labelled
    C - a list of classification labels
    L - a list of labeller IDs
    '''
    N = float(np.unique(U).shape[0])
    Uids = np.unique(U)
    
    Dobs = 0.0
    Dexpec = 0.0
    for i, u in enumerate(Uids):
        uidxs = U==u
        Lu = L[uidxs]
        m_u = Lu.shape[0]
        
        if m_u < 2:
            continue
        
        Cu = C[uidxs]
        
        #for cuj in Cu:
        #    Dobs += 1.0 / (m_u - 1.0) * np.sum(np.abs(cuj - Cu))
        
        Dobs += 1.0 / (m_u - 1.0) * np.sum(np.abs(Cu[:, np.newaxis] - Cu[np.newaxis, :]))

    # too much memory required            
    # Dexpec = np.sum(np.abs(C.flatten()[:, np.newaxis] - C.flatten()[np.newaxis, :]))
            
    for i in range(len(U)):
        if np.sum(U==U[i]) < 2:
            continue
        Dexpec += np.sum(np.abs(C[i] - C)) # sum up all differences regardless of user and data unit
        
    Dobs = 1 / N * Dobs
    Dexpec = Dexpec / (N * (N-1))  
    
    alpha = 1 - Dobs / Dexpec
    return alpha