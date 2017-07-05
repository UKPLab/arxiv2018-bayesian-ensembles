'''
Created on Nov 2, 2016

@author: Melvin Laux
'''
from sklearn.neighbors.kde import KernelDensity
import src.data.data_utils as dut
import numpy as np
from scipy.optimize import basinhopping          
            
class Clustering(object):
    '''
    classdocs
    '''

    ground_truth = None
    annotations = None
    num_docs = None
    doc_length = None
    
    gt_spans = None
    doc_anno = None
    
    curent_doc = None

    def __init__(self, ground_truth, annotations):
        '''
        Constructor
        '''
        
        self.ground_truth = ground_truth
        self.annotations = annotations
        
        self.anno_binary = np.where(annotations != 1)[0]
        
        self.anno_binary = self.anno_binary.reshape(-1, 1)
        
        self.num_docs = int(ground_truth[-1,0]+1)
        self.doc_length = int(ground_truth.shape[0]/self.num_docs)
        
        self.gt_spans = dut.iob_to_span(ground_truth, self.num_docs)
        self.anno_spans = dut.iob_to_span(annotations, self.num_docs)
        
        #print self.ground_truth
        #print self.num_docs
        
        
    def run(self):
        
        result = np.zeros((0,1))
        
        for doc in xrange(self.num_docs):
            
            self.curent_doc = doc
            self.doc_anno = np.extract(np.logical_and(self.anno_binary >= doc*self.doc_length, self.anno_binary <= (doc+1)*self.doc_length), self.anno_binary)[:,None]
        
            # function to minimise: negative data-log-likelihood
            f = lambda x: -self.calc_data_likelihood(x)
        
            # minimise bandwidth
            opt_h = basinhopping(f, np.log(0.5), stepsize=np.log(0.1), disp=False)
        
            # build model
            kde = KernelDensity(np.exp(opt_h.x), kernel='gaussian').fit(self.doc_anno)

            # error function to minimise: mean difference of annotation lengths
            g = lambda x: self.mean_length_diff(kde, x)
        
            opt_t = basinhopping(g, 0.02, stepsize=0.001, disp=False)
        
            result = np.append(result, self.label_tokens(kde, opt_t.x), axis=0)
            
        return result
    
    
    def calc_data_likelihood(self, lnBandwidth):
        
        h = np.exp(lnBandwidth)
        if h <= 0.1: 
            return -np.inf
                    
        #score = 0.0
        
        #for doc in xrange(int(self.num_docs)):
        
        kde = KernelDensity(bandwidth=h, kernel='gaussian').fit(self.doc_anno)
        #score += kde.score(self.anno_binary)
            
            
        return kde.score(self.doc_anno)
    
    
    def label_tokens(self, kde, threshold, return_spans=False):
        
        result = np.zeros((self.doc_length,1))
        
        scores = np.exp(kde.score_samples(np.arange(self.doc_length).reshape(-1,1)+self.curent_doc*self.doc_length))
        
        result[np.where(scores <= threshold)] = 1
        
        result[1:][((result == 0)[1:]) & ((result == 1)[:-1])] = 2
        
        if result[0]==0:  
            result[0] = 2 
            
        #for token in xrange(int(self.doc_length)):
            
        #    score = np.exp(kde.score_samples(np.array([token+self.curent_doc*self.doc_length]).reshape(-1, 1)))
            
        #    if score >= threshold:
        #        if result[token-1]==1 or token==0:
        #            result[token] = 2
        #        else:
        #            result[token] = 0
        #    else:
        #        result[token] = 1
        
        if return_spans:
            return dut.iob_to_span(result, 1)
        
        return result
    
    
    def mean_length_diff(self, kde, threshold):
        spans = self.label_tokens(kde, threshold, return_spans=True)[:,-1]
        
        if spans.size == 0:
            return np.inf
        
        return np.abs(np.mean(self.gt_spans[:,-1]) - np.mean(spans))
        