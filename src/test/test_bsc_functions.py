'''
Created on Nov 29, 2016

@author: Melvin Laux
'''

import unittest
import numpy as np
from joblib import Parallel

from src.baselines import ibcc
from src.bayesian_combination import bayesian_combination
from bayesian_combination.annotator_models import seq
from src.data.data_generator import DataGenerator

    
class Test(unittest.TestCase):
    def test_expec_lnB(self):
        
        # test with random input values where nu0 is of high magnitude (~1000)
        for _ in range(10):
            E_t = np.random.random_sample((5,3)) * 1e5
            nu0 = np.random.random_sample((3,)) * 1e5
            lnB, nu = ibcc._calc_q_A(E_t, nu0)
            
            #print nu/np.sum(nu), np.exp(lnB)
        
            assert np.allclose(nu/np.sum(nu), np.exp(lnB), atol=1e-3)
            
        # test with known specific values
        A = np.random.dirichlet(np.ones(10), 2).T / 2
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnB, nu = ibcc._calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, .5, atol=1e-10)
        assert np.allclose(lnB, -2 * np.log(2), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), 4).T / 4
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnB, nu = ibcc._calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, .25, atol=1e-10)
        assert np.allclose(lnB, -np.pi/2 - 3 * np.log(2), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), 3).T / 3
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        lnB, nu = ibcc._calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, 1.0/3, atol=1e-10)
        assert np.allclose(lnB,-np.pi/(2*np.sqrt(3)) - ((3.0/2.0) * np.log(3)), atol=1e-10)
        
        # 2 dimensional tests
        A = np.random.dirichlet(np.ones(10), (2,2)).T / 2
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnB, nu = ibcc._calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, 0.5, atol=1e-10)
        assert np.allclose(lnB, -2 * np.log(2), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), (3,3)).T / 3
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnB, nu = ibcc._calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, 1.0/3, atol=1e-10)
        assert np.allclose(lnB, -np.pi/(2*np.sqrt(3)) - ((3.0/2.0) * np.log(3)), atol=1e-10)
        
        A = np.random.dirichlet(np.ones(10), (4,4)).T / 4
        
        E_t = A[0:-1,:]
        nu0 = A[-1,:]
        
        lnB, nu = ibcc._calc_q_A(E_t, nu0)
        
        assert np.allclose(nu, .25, atol=1e-10)
        assert np.allclose(lnB, -np.pi/2 - 3 * np.log(2), atol=1e-10)
        
    def test_expec_lnPi(self):
        
        alpha = np.zeros((2,2,3,1))
        alpha[0,:,:,0] = [[2,1,2],[2,1,2]]
        alpha[1,:,:,0] = [[1,2,1],[1,2,1]]
        
        lnPi = seq.SequentialAnnotator._calc_q_pi(alpha)
        
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
        lnLambda = np.log(np.ones((10,4)) * (np.array(list(range(4))) + 1)[:, None].T)
        
        result = bayesian_combination._expec_t(lnR_, lnLambda)
        
        # ensure all rows sum up to 1
        assert np.allclose(np.sum(result,1), 1, atol=1e-10)
        
        # test values
        assert np.allclose(result[:,0], .1, atol=1e-10)
        assert np.allclose(result[:,1], .2, atol=1e-10)
        assert np.allclose(result[:,2], .3, atol=1e-10)
        assert np.allclose(result[:,3], .4, atol=1e-10)
        
        
    def test_expec_joint_t(self):
        
        T = 3
        
        lnR_ = np.log(np.ones((T,3)) * (np.array(list(range(3))) + 1)[:, None].T)
        lnLambda = np.log(np.ones((T,3))) 
        lnB = np.log(np.ones((4,3)))
        lnPi = np.log(np.ones((3,3,4,1)))
        C = np.ones((T,1), dtype=int)
        doc_start = np.array([1,0,0])

        result = bayesian_combination._expec_joint_t(lnR_, lnLambda, lnB, lnPi, None, C, None, doc_start, 3, seq.SequentialAnnotator)
        
        # ensure all rows sum up to 1
        assert np.allclose(np.sum(np.sum(result,-1),-1),1, atol=1e-10)
        
        print(result[0,:,:])
        
        # test whether flags set the correct values to zero 
        assert np.all(result[0,:-1,:] == 0)
        assert np.all(result[1,-1,:] == 0)
        assert np.all(result[2,-1,:] == 0)
        
    def test_forward_pass(self):
        initProbs = np.log(np.array([.5,.5]))
        lnB = np.log(np.array([[.7,.3],[.3,.7]]))
        lnPi = np.log(np.repeat(np.array([[.9,.1],[.2,.8]])[:,:,None,None],2,axis=2))
        
        C = np.ones((10,1), dtype=int)
        C[[2,7],0] = 2
                
        doc_start = np.array([1,0,0,0,0,1,0,0,0,0])

        with Parallel(n_jobs=-1) as parallel:
            lnR_ = bayesian_combination._parallel_forward_pass(parallel, C, None, lnB, lnPi, None, initProbs, doc_start, 2,
                                                               seq.SequentialAnnotator, skip=False)
        
        r_ = np.exp(lnR_)
        
        s_ = r_/np.sum(r_,1)[:,np.newaxis]
               
        target = np.array([[.8182,.1818],[.8834,.1166],[.1907,.8093],[.7308,.2692],[.8673,.1327],[.8182,.1818],[.8834,.1166],[.1907,.8093],[.7308,.2692],[.8673,.1327]])
        
        assert np.allclose(s_,target,atol=1e-4)
        
    def test_backward_pass(self):
        lnB = np.log(np.array([[.7,.3],[.3,.7]]))
        lnPi = np.log(np.repeat(np.array([[.9,.1],[.2,.8]])[:,:,None,None],2,axis=2))
        
        C = np.ones((10,1), dtype=int)
        C[[2,7],0] = 2
        
        doc_start = np.array([1,0,0,0,0,1,0,0,0,0])

        A = seq.SequentialAnnotator

        with Parallel(n_jobs=-1) as parallel:
            lnLambd = bayesian_combination._parallel_backward_pass(parallel, C, None, lnB, lnPi, None, doc_start, 2, A, skip=False)
        
            lambd = np.exp(lnLambd)
        
            lambdNorm = lambd/np.sum(lambd,1)[:,np.newaxis]
        
        target = np.array([
                         [0.5923176, 0.4076824],
                         [0.37626718, 0.62373282],
                         [0.65334282, 0.34665718],
                         [0.62727273, 0.37272727],
                         [0.5,        0.5       ],
                         [0.5923176,  0.4076824 ],
                         [0.37626718, 0.62373282],
                         [0.65334282, 0.34665718],
                         [0.62727273, 0.37272727],
                         [0.5,        0.5       ]])
        
        assert np.allclose(lambdNorm, target, atol=1e-4)
        
    def test_bsc(self):
        C = np.ones((5,10)) 
        
        doc_start = np.zeros((5,1))
        doc_start[[0,2]] = 1
        
        myBac = bayesian_combination.BC(L=3, K=10)
        
        result, _ =  myBac.fit_predict(C, doc_start)
        target = np.zeros((5,3))
        target[:,1] = 1
        
        assert np.allclose(target, result, atol=1e-4)
        
    def test_bsc_2(self):

        generator = DataGenerator('./config/data.ini', seed=42)
        
        generator.init_crowd_models(10, 0, 0)
        gt, C, _ = generator.generate_dataset(num_docs=2, doc_length=10, group_sizes=10, save_to_file=False)
        
        doc_start = np.zeros((20,1))
        doc_start[[0,10]] = 1
        
        myBac = bayesian_combination.BC(L=3, K=10)
        
        _, result = myBac.fit_predict(C, doc_start)
        
        assert np.allclose(gt.flatten(), result, atol=1e-4)
        
        
    def test_post_alpha(self):
        
        E_t = np.repeat(np.array([0.5, 0.5])[None,:], 8, axis=0)
        E_t[2,:] = np.array([1,0])
        E_t[6,:] = np.array([0,1])
        C = np.array([[2],[2],[2],[1],[1],[1],[1],[2]])
        doc_start = np.array([1,0,0,0,1,0,0,0])[:,None]
        alpha0 = np.zeros((2,2,3,1))
        alpha0[1,0,0,0] = 5
        alpha0[1,1,1,0] = 3

        result = seq.SequentialAnnotator.update_alpha(E_t, C, alpha0, None, doc_start, 2)
        
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
