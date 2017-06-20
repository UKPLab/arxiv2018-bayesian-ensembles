'''
Created on Nov 2, 2016

@author: Melvin Laux
'''
from sklearn.neighbors.kde import KernelDensity
import src.data.data_utils as dut
import numpy as np
import scipy.ndimage.filters as filters
from scipy.optimize import minimize, basinhopping
            
def find_peaks(estimator, max_val, step, threshold):
    f = lambda x,y: np.exp(estimator.score_samples(np.array([[x,y]])))

    x = np.arange(0, max_val, step)
    y = np.arange(0, max_val, step)
    X, Y = np.meshgrid(x, y)
    zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
        
    data_max = filters.maximum_filter(Z, (1,1))
    maxima = (Z == data_max)
    data_min = filters.minimum_filter(Z, (1,1))
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
        
    peaks = np.array([np.where(maxima==1)[1],np.where(maxima==1)[0]]).T*0.5
        
    max_vals = np.zeros((peaks.shape[0],1))
        
    for i in xrange(peaks.shape[0]):
        max_vals[i] = f(peaks[i,0], peaks[i,1])
        
    peaks = np.append(np.rint(peaks), max_vals, 1)
    peaks = peaks[peaks[:,2].argsort()[::-1]]
    
    #print 'peaks:', peaks
        
    peaks = dut.remove_span_overlaps(peaks.astype(int))
    
    return peaks        
            
            
class Clustering(object):
    '''
    classdocs
    '''

    ground_truth = None
    annotations = None
    num_docs = None
    doc_length = None
    
    gt_spans = None

    def __init__(self, ground_truth, annotations):
        '''
        Constructor
        '''
        
        self.ground_truth = ground_truth
        self.annotations = annotations
        
        self.anno_binary = np.where(annotations != 1)[0]
        
        self.anno_binary = self.anno_binary.reshape(-1, 1)
        
        self.num_docs = ground_truth[-1,0]+1
        self.doc_length = int(ground_truth.shape[0]/self.num_docs)
        
        self.gt_spans = dut.iob_to_span(ground_truth, self.num_docs)
        self.anno_spans = dut.iob_to_span(annotations, self.num_docs)
        #print self.ground_truth
        #print self.num_docs
        
        
    def run(self):
        
        # build function to minimise: negative data-log-likelihood
        f = lambda x: -self.calc_data_likelihood(x)
        
        # build optimisation constraint: bandwidth must be non-negative
        #cons = {'type':'ineq', 'fun': lambda x: x - 1e-5}
        
        #opt_h = minimize(f, np.log(0.5),  options={'disp': True})#, constraints=cons)
        opt_h = basinhopping(f,np.log(0.5), stepsize=np.log(0.1), disp=False)
        
        print 'opt_h:', np.exp(opt_h.x)
        
        x = self.anno_spans[:,1:] 
        
        kde = KernelDensity(np.exp(opt_h.x), kernel='gaussian').fit(self.anno_binary)

        #g = lambda x: np.abs(np.mean(self.spans[:,-1]) - np.mean(self.label_tokens(kde, x, return_spans=True)[:,-1]))
        g = lambda x: self.mean_length_diff(kde, x)
        
        opt_t = basinhopping(g,0.02,stepsize=0.001, disp=False)#  options={'disp': True})
        
        print 'opt_t:', opt_t.x
        
        #print self.label_tokens(kde, opt_t.x, return_spans=True)[:,1:], self.gt_spans[:,1:]
        
        return self.label_tokens(kde, opt_t.x)

        
    def run_kde(self, bandwidth, threshold, result_spans=False):
        
        result = None
        
        for doc in xrange(int(self.num_docs)):
            x = self.anno_spans[self.ground_truth[:,0]==doc][:,1:] 
            
            #print x
            kde = KernelDensity(np.exp(bandwidth), kernel='gaussian').fit(x)
            
            peaks = find_peaks(kde, self.doc_length, 0.25, threshold)
            
            if result_spans:
                
                print peaks 
                
                if result is None:
                    result = peaks
                else:
                    result = np.append(result, peaks, 0)
            
            else:
                iob = dut.span_to_iob(peaks[:,], self.num_docs, self.doc_length)
            
                if result is None:
                    result = iob
                else:
                    result = np.append(result, iob, 0)
        
        return result
    
    def calc_data_likelihood(self, lnBandwidth):
        
        h = np.exp(lnBandwidth)
        if h <= 0.1: 
            return -np.inf
            
        #print h, h <= 0
        
        score = 0.0
        
        for doc in xrange(int(self.num_docs)):
            x = self.anno_spans [self.anno_spans[:,0]==doc][:,1:] 
            kde = KernelDensity(bandwidth=h, kernel='gaussian').fit(self.anno_binary)
            score += kde.score(self.anno_binary)
            
            
        return score
    
    
    def label_tokens(self, kde, threshold, return_spans=False):
        
        result = np.zeros((self.doc_length,1))
                
        for token in xrange(int(self.doc_length)):
            
            score = np.exp(kde.score_samples(np.array([token]).reshape(-1, 1)))
            
            if score >= threshold:
                if result[token-1]==1 or token==0:
                    result[token] = 2
                else:
                    result[token] = 0
            else:
                result[token] = 1
        
        if return_spans:
            return dut.iob_to_span(result, 1)
        
        return result
    
    
    def mean_length_diff(self, kde, threshold):
        spans = self.label_tokens(kde, threshold, return_spans=True)[:,-1]
        
        if spans.size == 0:
            return np.inf
        
        #print 'spans not empty'
        diff = np.abs(np.mean(self.gt_spans[:,-1]) - np.mean(spans))
        #print diff, threshold
        return diff
        