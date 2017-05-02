'''
Created on Nov 29, 2016

@author: Melvin Laux
'''

import unittest
import numpy as np
from src.baselines import ibcc 
from src.algorithm.expectations import expec_t,expec_joint_t
from src.algorithm import bac

    
class Test(unittest.TestCase):
    def test_expec_lnkappa(self):
        
        # test with random input values where nu0 is of high magnitude (~1000)
        for _ in xrange(10):
            E_t = np.random.random_sample((5,3)) * 1e5
            nu0 = np.random.random_sample((3,)) * 1e5
            lnkappa, nu = ibcc.calc_q_A(E_t, nu0)
            
            #print nu/np.sum(nu), np.exp(lnkappa)
        
            assert np.allclose(nu/np.sum(nu), np.exp(lnkappa), atol=1e-3)
            
        # test with known specific values
        A = np.random.dirichlet(np.ones(10), 2).T / 2
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnkappa, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, .5, atol=1e-10)
        assert np.allclose(lnkappa, -2 * np.log(2), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), 4).T / 4
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnkappa, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, .25, atol=1e-10)
        assert np.allclose(lnkappa, -np.pi/2 - 3 * np.log(2), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), 3).T / 3
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        lnkappa, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, 1.0/3, atol=1e-10)
        assert np.allclose(lnkappa,-np.pi/(2*np.sqrt(3)) - ((3.0/2.0) * np.log(3)), atol=1e-10)
        
        # 2 dimensional tests
        A = np.random.dirichlet(np.ones(10), (2,2)).T / 2
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnkappa, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, 0.5, atol=1e-10)
        assert np.allclose(lnkappa, -2 * np.log(2), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), (3,3)).T / 3
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnkappa, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, 1.0/3, atol=1e-10)
        assert np.allclose(lnkappa, -np.pi/(2*np.sqrt(3)) - ((3.0/2.0) * np.log(3)), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), (4,4)).T / 4
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnkappa, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, .25, atol=1e-10)
        assert np.allclose(lnkappa, -np.pi/2 - 3 * np.log(2), atol=1e-10)
        
    def test_expec_lnPi(self):
        
        alpha = np.zeros((2,2,3,1))
        alpha[0,:,:,0] = [[2,1,2],[2,1,2]]
        alpha[1,:,:,0] = [[1,2,1],[1,2,1]]
        
        lnPi = bac.calc_q_pi(alpha)
        
        # print lnPi[0,:,:,0]
        # print lnPi[1,:,:,0]
        
        assert lnPi[0,0,0,0] > lnPi[0,0,1,0]
        assert lnPi[0,0,2,0] > lnPi[0,0,1,0]
        assert lnPi[0,1,0,0] > lnPi[0,1,1,0]
        assert lnPi[0,1,2,0] > lnPi[0,1,1,0]
        
        assert lnPi[1,0,0,0] < lnPi[1,0,1,0]
        assert lnPi[1,0,2,0] < lnPi[1,0,1,0]
        assert lnPi[1,1,0,0] < lnPi[1,1,1,0]
        assert lnPi[1,1,2,0] < lnPi[1,1,1,0]
        
        
    def test_expec_t_trans(self):
        
        E_t = np.array([np.ones(2),np.zeros(2)]).T
        E_t_goal = np.zeros((2,2,2))
        E_t_goal[:,0,0] = 1
        E_t_goal[0,1,0] = 1
        
        E_t_trans = ibcc.expec_t_trans(E_t)
        
        assert np.allclose(E_t_goal, E_t_trans)
        
    
    def test_expec_t(self): 
        
        lnR_ = np.log(np.ones((10,4))) 
        lnLambda = np.log(np.ones((10,4)) * (np.array(range(4)) + 1)[:, None].T)
        
        result = expec_t(lnR_, lnLambda)  
        
        # ensure all rows sum up to 1
        assert np.allclose(np.sum(result,1), 1, atol=1e-10)
        
        # test values
        assert np.allclose(result[:,0], .1, atol=1e-10)
        assert np.allclose(result[:,1], .2, atol=1e-10)
        assert np.allclose(result[:,2], .3, atol=1e-10)
        assert np.allclose(result[:,3], .4, atol=1e-10)
        
        
    def test_expec_joint_t(self):
        
        T = 3
        
        lnR_ = np.log(np.ones((T,3)) * (np.array(range(3)) + 1)[:, None].T)
        lnLambda = np.log(np.ones((T,3))) 
        lnA = np.log(np.ones((4,3)))
        lnPi = np.log(np.ones((3,3,4,1)))
        C = np.ones((T,1)) 
        doc_start = np.array([1,0,0])
        
        result = expec_joint_t(lnR_, lnLambda, lnA, lnPi, C, doc_start)
        
        # ensure all rows sum up to 1
        assert np.allclose(np.sum(np.sum(result,-1),-1),1, atol=1e-10)
        
        assert np.all(result[0,:-1,:] == 0)
        assert np.all(result[1,-1,:] == 0)
        assert np.all(result[2,-1,:] == 0)
        
        
              
if __name__ == "__main__":
    unittest.main()
