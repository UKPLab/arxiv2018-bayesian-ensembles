'''
Created on Nov 2, 2016

@author: Melvin Laux
'''
from sklearn.neighbors.kde import KernelDensity
import data.data_utils as dut
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from data.data_utils import iob_to_span

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
        
        
    def run_kde(self, bandwidth=1.0, threshold=0.001, precision=0.25):
        
        result = None
        
        for doc in xrange(int(self.num_docs)):
            x = self.spans[self.spans[:,0]==doc][:,1:] 
            kde = KernelDensity(bandwidth, kernel='gaussian').fit(x)
        
            f = lambda x,y: np.exp(kde.score_samples(np.array([[x,y]])))

            x = np.arange(0, self.doc_length, precision)
            y = np.arange(0, self.doc_length, precision)
            X, Y = np.meshgrid(x, y)
            zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
            Z = zs.reshape(X.shape)
        
            data_max = filters.maximum_filter(Z, (3,3))
            maxima = (Z == data_max)
            data_min = filters.minimum_filter(Z, (3,3))
            diff = ((data_max - data_min) > threshold)
            maxima[diff == 0] = 0
        
            peaks = np.array([np.where(maxima==1)[1],np.where(maxima==1)[0]]).T*precision
        
            max_vals = np.zeros((peaks.shape[0],1))
        
            for i in xrange(peaks.shape[0]):
                max_vals[i] = f(peaks[i,0], peaks[i,1])
        
            peaks = np.append(np.rint(peaks), max_vals, 1)
            peaks = peaks[peaks[:,2].argsort()[::-1]]
        
            peaks = dut.remove_span_overlaps(peaks.astype(int))
            
            iob = dut.span_to_iob(peaks[:,], self.num_docs, self.doc_length)
            
            
            if result == None:
                result = iob
            else:
                result = np.append(result, iob, 0)
        
        return result
    
        