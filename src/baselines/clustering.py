'''
Created on Nov 2, 2016

@author: Melvin Laux
'''
from sklearn.neighbors.kde import KernelDensity
import sklearn.metrics as skm
import data.data_utils as dut
import numpy as np
import scipy.ndimage.filters as filters
from scipy.optimize import minimize
from sklearn.cross_validation import cross_val_score
            
def find_peaks(estimator, max_val, step, threshold):
    f = lambda x,y: np.exp(estimator.score_samples(np.array([[x,y]])))

    x = np.arange(0, max_val, step)
    y = np.arange(0, max_val, step)
    X, Y = np.meshgrid(x, y)
    zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
        
    data_max = filters.maximum_filter(Z, (3,3))
    maxima = (Z == data_max)
    data_min = filters.minimum_filter(Z, (3,3))
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
        
    peaks = np.array([np.where(maxima==1)[1],np.where(maxima==1)[0]]).T*0.5
        
    max_vals = np.zeros((peaks.shape[0],1))
        
    for i in xrange(peaks.shape[0]):
        max_vals[i] = f(peaks[i,0], peaks[i,1])
        
    peaks = np.append(np.rint(peaks), max_vals, 1)
    peaks = peaks[peaks[:,2].argsort()[::-1]]
        
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
    
    spans = None

    def __init__(self, ground_truth, annotations):
        '''
        Constructor
        '''
        
        self.ground_truth = ground_truth
        self.annotations = annotations
        
        self.num_docs = ground_truth[-1,0]+1
        self.doc_length = ground_truth.shape[0]/self.num_docs
        
        self.spans = dut.iob_to_span(self.annotations, self.num_docs)
        
        
    def run(self):
        # build function to minimise: negative f1-score of clustering results
        f = lambda x: -skm.f1_score(self.run_kde(x[0], x[1]), self.ground_truth[:,1], average='macro')
        
        # build optimisation constraint: bandwidth must be non-negative
        cons = {'type':'ineq', 'fun':lambda x: x[0]}
        
        opt = minimize(f, np.array([1.0, 0.001]), constraints=cons)
        
        return self.run_kde(opt.x[0], opt.x[1])

        
    def run_kde(self, bandwidth, threshold):
        
        result = None
        
        for doc in xrange(int(self.num_docs)):
            x = self.spans[self.spans[:,0]==doc][:,1:] 
            
            #print x
            kde = KernelDensity(bandwidth, kernel='gaussian').fit(x)
            
            peaks = find_peaks(kde, self.doc_length, 0.5, threshold)
            
            iob = dut.span_to_iob(peaks[:,], self.num_docs, self.doc_length)
            
            
            if result is None:
                result = iob
            else:
                result = np.append(result, iob, 0)
        
        return result
    
        