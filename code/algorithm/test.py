'''
Created on Jan 17, 2017

@author: wiluqa
'''
import unittest
import numpy as np
from algorithm.forward_backward import forward_pass, backward_pass
from sklearn.preprocessing.data import normalize
from numpy import allclose


class Test(unittest.TestCase):

    def test_forward_pass(self):
        pi = np.array([.5,.5])
        A = np.array([[.7,.3],[.3,.7]])
        C = np.array([[.9,.1],[.2,.8]])
        y = np.array([0,0,1,0,0])
        
        a = forward_pass(y, A, C, pi)
        
        target = np.array([[.8182,.1818],[.8834,.1166],[.1907,.8093],[.7308,.2692],[.8673,.1327]])
        
        assert allclose(a,target,atol=1e-4)
        
    def test_backward_pass(self):
        pi = np.array([.5,.5])
        A = np.array([[.7,.3],[.3,.7]])
        C = np.array([[.9,.1],[.2,.8]])
        y = np.array([0,0,1,0,0])
        
        b = backward_pass(y, A, C, pi)
        
        target = np.array([[.6469,.3531],[.5923,.4077],[.3763,.6237],[.6533,.3467],[.6273,.3727],[1,1]])
        
        assert allclose(b,target,atol=1e-4)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testFB']
    unittest.main()