'''
Created on Nov 29, 2016

@author: Melvin Laux
'''

import unittest
import numpy as np
from src.baselines import ibcc 
from src.algorithm import bac

    
class Test(unittest.TestCase):
    def test_expec_lnA(self):
        
        # test with random input values where nu0 is of high magnitude (~1000)
        for _ in xrange(10):
            E_t = np.random.random_sample((5,3)) * 1e5
            nu0 = np.random.random_sample((3,)) * 1e5
            lnA, nu = ibcc.calc_q_A(E_t, nu0)
            
            #print nu/np.sum(nu), np.exp(lnA)
        
            assert np.allclose(nu/np.sum(nu), np.exp(lnA), atol=1e-3)
            
        # test with known specific values
        A = np.random.dirichlet(np.ones(10), 2).T / 2
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnA, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, .5, atol=1e-10)
        assert np.allclose(lnA, -2 * np.log(2), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), 4).T / 4
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnA, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, .25, atol=1e-10)
        assert np.allclose(lnA, -np.pi/2 - 3 * np.log(2), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), 3).T / 3
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        lnA, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, 1.0/3, atol=1e-10)
        assert np.allclose(lnA,-np.pi/(2*np.sqrt(3)) - ((3.0/2.0) * np.log(3)), atol=1e-10)
        
        # 2 dimensional tests
        A = np.random.dirichlet(np.ones(10), (2,2)).T / 2
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnA, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, 0.5, atol=1e-10)
        assert np.allclose(lnA, -2 * np.log(2), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), (3,3)).T / 3
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnA, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, 1.0/3, atol=1e-10)
        assert np.allclose(lnA, -np.pi/(2*np.sqrt(3)) - ((3.0/2.0) * np.log(3)), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), (4,4)).T / 4
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnA, nu = ibcc.calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, .25, atol=1e-10)
        assert np.allclose(lnA, -np.pi/2 - 3 * np.log(2), atol=1e-10)
        
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
        
        result = bac.expec_t(lnR_, lnLambda)  
        
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
        
        result = bac.expec_joint_t(lnR_, lnLambda, lnA, lnPi, C, doc_start)
        
        # ensure all rows sum up to 1
        assert np.allclose(np.sum(np.sum(result,-1),-1),1, atol=1e-10)
        
        assert np.all(result[0,:-1,:] == 0)
        assert np.all(result[1,-1,:] == 0)
        assert np.all(result[2,-1,:] == 0)
        
    def test_forward_pass(self):
        initProbs = np.log(np.array([.5,.5]))
        lnA = np.log(np.array([[.7,.3],[.3,.7]]))
        lnPi = np.log(np.repeat(np.array([[.9,.1],[.2,.8]])[:,:,np.newaxis,np.newaxis],2,axis=2))
        
        C = np.ones((10,1))
        C[[2,7],0] = 2
                
        doc_start = np.array([1,0,0,0,0,1,0,0,0,0])
        
        lnR_ = bac.forward_pass(C, lnA, lnPi, initProbs, doc_start, skip=False)
        
        r_ = np.exp(lnR_)
        
        s_ = r_/np.sum(r_,1)[:,np.newaxis]
               
        target = np.array([[.8182,.1818],[.8834,.1166],[.1907,.8093],[.7308,.2692],[.8673,.1327],[.8182,.1818],[.8834,.1166],[.1907,.8093],[.7308,.2692],[.8673,.1327]])
        
        assert np.allclose(s_,target,atol=1e-4)
        
    def test_backward_pass(self):
        lnA = np.log(np.array([[.7,.3],[.3,.7]]))
        lnPi = np.log(np.repeat(np.array([[.9,.1],[.2,.8]])[:,:,np.newaxis,np.newaxis],2,axis=2))
        
        C = np.ones((10,1))
        C[[2,7],0] = 2
        
        doc_start = np.array([1,0,0,0,0,1,0,0,0,0])
        
        lnLambd = bac.backward_pass(C, lnA, lnPi, doc_start, skip=False)
        
        lambd = np.exp(lnLambd)
        
        lambdNorm = lambd/np.sum(lambd,1)[:,np.newaxis]
        
        target = np.array([[.6469,.3531],[.5923,.4077],[.3763,.6237],[.6533,.3467],[.6273,.3727],[.6469,.3531],[.5923,.4077],[.3763,.6237],[.6533,.3467],[.6273,.3727]])
        
        assert np.allclose(lambdNorm,target,atol=1e-4)
        
    def test_bac(self):
        C = np.ones((5,10)) * 2
        
        doc_start = np.zeros((5,1))
        doc_start[[0,2]] = 1
        
        myBac = bac.BAC(L=3, K=10)
        
        result =  myBac.run(C, doc_start)
        target = np.zeros((5,3))
        target[:,1] = 1
        
        assert np.allclose(target, result, atol=1e-4)
        
    def test_bac_2(self):
        C = np.genfromtxt('../src/output/data/annotations.csv', delimiter=',') + 1
        gt = np.genfromtxt('../src/output/data/ground_truth.csv', delimiter=',')[:,1] + 1
        
        doc_start = np.zeros((20,1))
        doc_start[[0,10]] = 1
        
        myBac = bac.BAC(L=3, K=10)
        
        result = myBac.run(C, doc_start)
        
        target = np.zeros((20,3))
        
        for t in xrange(20):
            target[t,int(gt[t]-1)] = 1
        
        assert np.allclose(target, result, atol=1e-4)
        
        
    def test_post_alpha(self):
        
        E_t = np.repeat(np.array([0.5, 0.5])[None,:], 8, axis=0)
        E_t[2,:] = np.array([1,0])
        E_t[6,:] = np.array([0,1])
        C = np.array([[2],[2],[2],[1],[1],[1],[1],[2]])
        doc_start = np.array([1,0,0,0,1,0,0,0])[:,None]
        alpha0 = np.zeros((2,2,3,1))
        alpha0[1,0,0,0] = 5
        alpha0[1,1,1,0] = 3
        
        result = bac.post_alpha(E_t, C, alpha0, doc_start)
        
        target = np.ones((2,2,3,1)) * .5
        
        target[1,0,0,0] = 6.5
        target[1,1,1,0] = 3.5
        target[0,1,1,0] = 1.5
        
        # all values should be non-negative
        assert np.all(result >= 0)
        # check values
        assert np.all(result==target)
              
if __name__ == "__main__":
    unittest.main()
