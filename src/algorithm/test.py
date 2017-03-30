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
        initProbs = np.array([.5,.5])
        lnA = np.log(np.array([[.7,.3],[.3,.7]]))
        lnPi = np.log(np.repeat(np.array([[.9,.1],[.2,.8]])[:,:,np.newaxis,np.newaxis],2,axis=2))
        
        C = np.ones((10,1))
        C[[2,7],0] = 2
                
        doc_start = np.array([1,0,0,0,0,1,0,0,0,0])
        
        r_ = forward_pass(C, lnA, lnPi, initProbs, doc_start, skip=False)
        
        print r_
        
        target = np.array([[.8182,.1818],[.8834,.1166],[.1907,.8093],[.7308,.2692],[.8673,.1327],[.8182,.1818],[.8834,.1166],[.1907,.8093],[.7308,.2692],[.8673,.1327]])
        
        assert allclose(r_,target,atol=1e-4)
        
    def test_backward_pass(self):
        lnA = np.array([[.7,.3],[.3,.7]])
        lnPi = np.log(np.repeat(np.array([[.9,.1],[.2,.8]])[:,:,np.newaxis,np.newaxis],2,axis=2))
        
        C = np.ones((10,1))
        C[[2,7],0] = 2
        
        doc_start = np.array([1,0,0,0,0,1,0,0,0,0])
        
        betas = backward_pass(C, lnA, lnPi, doc_start, skip=False)
        
        print betas
        
        target = np.array([[.6469,.3531],[.5923,.4077],[.3763,.6237],[.6533,.3467],[.6273,.3727],[.6469,.3531],[.5923,.4077],[.3763,.6237],[.6533,.3467],[.6273,.3727]])
        
        assert allclose(betas,target,atol=1e-4)
        
    def test_bac(self):
        C = np.ones((5,10)) * 2
        
        doc_start = np.zeros((5,1))
        doc_start[[0,2]] = 1
        
        myBac = bac.BAC(L=3, T=5, K=10)
        
        print myBac.run(C, doc_start)


if __name__ == "__main__":
    unittest.main()