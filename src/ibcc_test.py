'''
Created on 8 Apr 2015

@author: edwin
'''
import unittest
from baselines import ibcc
import logging
import numpy as np

def check_accuracy(pT, target_acc, goldfile='./data/gold_verify.csv'):
    # check values are in tolerance range
    gold = np.genfromtxt(goldfile)
    decisions = np.round(pT[:,1])        
    errors = np.abs(gold-decisions)
    nerrors = np.nansum(errors)
    acc = np.round(1 - (nerrors/float(np.sum(np.isfinite(gold)))), decimals=5)
    logging.info( "accuracy is %f, nerrors=%i" % (acc, nerrors))
    assert acc==target_acc

def check_accuracy_multi(pT, target_acc, goldfile='./data/gold5_verify.csv'):
    # check values are in tolerance range
    gold = np.genfromtxt(goldfile)
    nerrors = 0
    for j in range(pT.shape[1]):
        decisions = np.round(pT[:,j])
        goldj = gold==j        
        errors = np.abs(goldj-decisions)
        errors = errors[goldj]
        nerrors += np.nansum(errors)
    acc = np.round(1 - (nerrors/float(np.sum(np.isfinite(gold)))), decimals=5)
    logging.info( "accuracy is %f, nerrors=%i" % (acc, nerrors))
    assert acc==target_acc
    
def check_outputsize(pT, combiner, shape=(2,2,5), ptlength=100):
    # check output has right number of data points
    assert pT.shape[0]==ptlength
    logging.info("Alpha shape: " + str(combiner.alpha.shape))
    assert combiner.alpha.shape == shape
    
class Test(unittest.TestCase):
    def testSparseList_noGold(self):
        configFile = './config/sparse_nogold.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner, ptlength=199)
        check_accuracy(pT, 0.82, goldfile='./data/gold_mixed_verify.csv')
         
    def testTable_noGold(self):
        # Crowdlabels contains some NaNs and some -1s.
        configFile = './config/table_nogold.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner)
        check_accuracy(pT, 0.82) 
        
    def testSparseList_withGold(self):
        #Gold labels is longer than the no. crowd-labelled data points
        configFile = './config/sparse_gold.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner)
        check_accuracy(pT, 0.95)
         
    def testTable_withGold(self):
        #Gold labels is longer than the no. crowd-labelled data points
        configFile = './config/table_gold.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner)
        check_accuracy(pT, 0.95) 
                
    def testSparseList_shortGold(self):
        #Gold labels is shorter than the no. crowd-labelled data points
        configFile = './config/sparse_shortgold.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner)
        check_accuracy(pT, 0.95)
        
    def testTable_shortGold(self):
        #Gold labels is shorter than the no. crowd-labelled data points
        configFile = './config/table_shortgold.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner)
        check_accuracy(pT, 0.95)     
      
    def testSparseList_shortGoldMatrix(self):
        #Gold labels is shorter than the no. crowd-labelled data points
        configFile = './config/sparse_shortgoldmat.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner, ptlength=199)
        check_accuracy(pT, 0.94, goldfile='./data/gold_mixed_verify.csv')
           
    def testTable_shortGoldMatrix(self):
        #Gold labels is shorter than the no. crowd-labelled data points
        configFile = './config/table_shortgoldmat.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner, ptlength=199)
        check_accuracy(pT, 0.94, goldfile='./data/gold_mixed_verify.csv')   
               
    def testSparseList_withGold_5classes(self):
        #Gold labels is longer than the no. crowd-labelled data points
        configFile = './config/sparse_gold5.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner, (5,5,5))
        check_accuracy_multi(pT, 1)
         
    def testTable_withGold_5classes(self):
        #Gold labels is longer than the no. crowd-labelled data points
        configFile = './config/table_gold5.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner, (5,5,5))
        check_accuracy_multi(pT, 1)           
    
    def testSparseList_lowerbound(self):
        #Gold labels is longer than the no. crowd-labelled data points
        configFile = './config/sparse_gold_lowerbound.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner)
        check_accuracy(pT, 0.95)
            
    def testTable_lowerbound_5classes(self):
        #Gold labels is longer than the no. crowd-labelled data points
        configFile = './config/table_gold5_lowerbound.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner, (5,5,5))
        check_accuracy_multi(pT, 1)    

        
# OPTIMIZATION --------------------------------------------------------------------------------------------------------

    def testSparseList_opt(self):
        configFile = './config/sparse_nogold.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None, optimise_hyperparams=True)
        check_outputsize(pT, combiner, ptlength=199)
        check_accuracy(pT, 0.82, goldfile='./data/gold_mixed_verify.csv')
            
    def testTable_withGold_5classes_opt(self):
        #Gold labels is longer than the no. crowd-labelled data points
        configFile = './config/table_gold5.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None, optimise_hyperparams=True)
        check_outputsize(pT, combiner, (5,5,5))
        check_accuracy_multi(pT, 1)      
          
# SCORES NOT FROM 0 ---------------------------------------------------------------------------------------------------
 
    def testSparseList_scores(self):
        configFile = './config/sparse_nogold_mixscores.py'
        pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=None)
        check_outputsize(pT, combiner, ptlength=199)
        check_accuracy(pT, 0.82, goldfile='./data/gold_mixed_verify.csv')

# SETUP ETC. ----------------------------------------------------------------------------------------------------------

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("TEST: "+self._testMethodName)

# TODO list -----------------------------------------------------------------------------------------------------------
# Add tests for when scores are not consecutive from 0

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testSparseList']
    unittest.main()