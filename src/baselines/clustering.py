'''
Created on Nov 2, 2016

@author: Melvin Laux
'''
from sklearn.neighbors.kde import KernelDensity
import data.data_utils as dut
import numpy as np
from scipy.optimize import basinhopping          
            
class Clustering(object):
    '''
    classdocs
    '''

    annotations = None
    num_docs = None
    doc_length = None
    
    doc_anno = None
    
    curent_doc = None
    
    doc_start = None

    def __init__(self, ground_truth, annotations, doc_start):
        '''
        Constructor
        '''
        
        self.annotations = annotations
        
        self.anno_binary = np.where(np.isin(annotations,[0,2]))[0]
        self.anno_binary = self.anno_binary.reshape(-1, 1)
        
        self.num_docs = int(np.sum(doc_start))
        
        self.anno_spans = dut.iob_to_span(annotations, self.num_docs)
        
        self.doc_start = doc_start

        
    def run(self):
        
        result = np.zeros((0,1))
        
        self.start_idxs = np.r_[np.where(self.doc_start==1)[0],len(self.doc_start)]
        self.doc_length = self.start_idxs[0]
        for doc in range(self.num_docs):
            
            self.curent_doc = doc
            
            self.doc_anno = np.extract(np.logical_and(self.anno_binary >= self.start_idxs[doc], self.anno_binary < self.start_idxs[doc+1]), self.anno_binary)[:,None]
            if doc < self.num_docs:
                self.doc_length = self.start_idxs[doc+1] - self.start_idxs[doc]
            else:
                self.doc_length = self.annotations.shape[0] - self.start_idxs[doc]
                
            if self.doc_anno.shape == (0,1):
                result = np.append(result,np.zeros((self.doc_length,1)), axis=0)
                continue
        
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
    
    '''
    Calculates the data log-likelihood of the model with a given log bandwidth.
    '''
    def calc_data_likelihood(self, lnBandwidth):
        
        # return negative infinity if bandwidth is too small
        h = np.exp(lnBandwidth)
        if h <= 0.1: 
            return -np.inf
        
        return KernelDensity(bandwidth=h, kernel='gaussian').fit(self.doc_anno).score(self.doc_anno)
    
    '''
    Label all tokens using the given kde and threshold. If the kde-score of a position is above this threshold the label at this position will be marked as 'inside'.
    Tokens marked as 'inside' where the previous label is marked 'outside' will be marked as 'beginning'. Output can be either in 'IOB'-format or in 'span'-format.
    '''
    def label_tokens(self, kde, threshold, return_spans=False):
        
        result = np.zeros((self.doc_length,1))
        
        scores = np.exp(kde.score_samples(np.arange(self.doc_length).reshape(-1,1)+self.curent_doc*self.doc_length))
        
        result[np.where(scores <= threshold)] = 1
        
        result[1:][((result == 0)[1:]) & ((result == 1)[:-1])] = 2
        
        if result[0]==0:  
            result[0] = 2 
        
        if return_spans:
            return dut.iob_to_span(result, 1)
        
        return result
    
    '''
    Calculates the difference between the mean lengths of the predicted sequence and the gold standard.
    '''
    def mean_length_diff(self, kde, threshold):
        spans = self.label_tokens(kde, threshold, return_spans=True)[:,-1]
        
        # set to infinity if no spans were predicted
        if spans.size == 0:
            return np.inf
        
        return np.abs(np.mean(self.anno_spans[:,-1]) - np.mean(spans))
        