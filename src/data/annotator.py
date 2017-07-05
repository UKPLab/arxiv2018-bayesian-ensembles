'''
Created on Oct 18, 2016

@author: Melvin Laux
'''

import numpy as np
import data_utils

class Annotator(object):
    '''
    classdocs
    '''
    
    model = None

    def __init__(self, model):
        '''
        Constructor
        '''
        self.model = model
        
    def annotate_word(self, prev_label, true_label):
        return np.asscalar(np.random.choice(range(3), 1, p = self.model[int(prev_label),int(true_label),:]))
    
    def annotate_document(self, document):
        
        # initialise annotation vector
        annotation = -np.ones(document.shape)
        
        while True:
        
            # set first label of document
            annotation[0] = self.annotate_word(3, document[0])
        
            for i in xrange(document.shape[0]):
                annotation[i] = self.annotate_word(annotation[i-1], document[i])
                
            if data_utils.check_document_syntax(annotation):
                break
    
        return annotation
    
    def annotate(self, ground_truth):
        
        data = np.ones((ground_truth.shape[0],))
            
        # iterate through documents
        for doc_id in xrange(int(ground_truth[-1,0]) + 1):
            data[ground_truth[:,0]==doc_id,] = self.annotate_document(ground_truth[ground_truth[:,0]==doc_id,1])
            
        return data