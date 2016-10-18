'''
Created on Oct 18, 2016

@author: Melvin Laux
'''

import numpy as np

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
        return np.asscalar(np.random.choice(range(3), 1, p = self.model[prev_label,true_label,:]))