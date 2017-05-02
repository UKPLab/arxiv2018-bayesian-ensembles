'''
Created on Jan 17, 2017

@author: Melvin Laux
'''
import unittest
import numpy as np
from src.algorithm.forward_backward import forward_pass, backward_pass
from src.algorithm import bac
from numpy import allclose

class Test(unittest.TestCase):

    def test_forward_pass(self):
        initProbs = np.log(np.array([.5,.5]))
        lnA = np.log(np.array([[.7,.3],[.3,.7]]))
        lnPi = np.log(np.repeat(np.array([[.9,.1],[.2,.8]])[:,:,np.newaxis,np.newaxis],2,axis=2))
        
        C = np.ones((10,1))
        C[[2,7],0] = 2
                
        doc_start = np.array([1,0,0,0,0,1,0,0,0,0])
        
        lnR_ = forward_pass(C, lnA, lnPi, initProbs, doc_start, skip=False)
        
        r_ = np.exp(lnR_)
        
        s_ = r_/np.sum(r_,1)[:,np.newaxis]
               
        target = np.array([[.8182,.1818],[.8834,.1166],[.1907,.8093],[.7308,.2692],[.8673,.1327],[.8182,.1818],[.8834,.1166],[.1907,.8093],[.7308,.2692],[.8673,.1327]])
        
        assert allclose(s_,target,atol=1e-4)
        
    def test_backward_pass(self):
        lnA = np.log(np.array([[.7,.3],[.3,.7]]))
        lnPi = np.log(np.repeat(np.array([[.9,.1],[.2,.8]])[:,:,np.newaxis,np.newaxis],2,axis=2))
        
        C = np.ones((10,1))
        C[[2,7],0] = 2
        
        doc_start = np.array([1,0,0,0,0,1,0,0,0,0])
        
        lnLambd = backward_pass(C, lnA, lnPi, doc_start, skip=False)
        
        lambd = np.exp(lnLambd)
        
        lambdNorm = lambd/np.sum(lambd,1)[:,np.newaxis]
        
        target = np.array([[.6469,.3531],[.5923,.4077],[.3763,.6237],[.6533,.3467],[.6273,.3727],[.6469,.3531],[.5923,.4077],[.3763,.6237],[.6533,.3467],[.6273,.3727]])
        
        assert allclose(lambdNorm,target,atol=1e-4)
        
    def test_bac(self):
        C = np.ones((5,10)) * 2
        
        doc_start = np.zeros((5,1))
        doc_start[[0,2]] = 1
        
        myBac = bac.BAC(L=3, T=5, K=10)
        
        result =  myBac.run(C, doc_start)
        target = np.zeros((5,3))
        target[:,1] = 1
        
        assert allclose(target, result, atol=1e-4)
        
    def test_bac_2(self):
        C = np.genfromtxt('../output/data/annotations.csv', delimiter=',') + 1
        gt = np.genfromtxt('../output/data/ground_truth.csv', delimiter=',')[:,1] + 1
        
        doc_start = np.zeros((20,1))
        doc_start[[0,10]] = 1
        
        myBac = bac.BAC(L=3, T=20, K=10)
        
        result = myBac.run(C, doc_start)
        
        target = np.zeros((20,3))
        
        for t in xrange(20):
            target[t,int(gt[t]-1)] = 1
        
        assert allclose(target, result, atol=1e-4)
        
        
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