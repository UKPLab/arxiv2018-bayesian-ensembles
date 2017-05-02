'''
Created on Oct 25, 2016

@author: Melvin Laux
'''

import numpy as np
import sklearn.metrics as metrics

class MajorityVoting(object):
    '''
    classdocs
    '''
    majority = None
    annotations = None
    
    accuracies = None
    
    votes = None
    
    num_labels = None
    num_annotators = None
    num_words = None

    def __init__(self, ground_truth, annotations, num_labels):
        '''
        Constructor
        '''
        
        self.ground_truth = ground_truth
        self.annotations = annotations
        self.num_words, self.num_annotators = annotations.shape
        self.num_labels = num_labels
        self.accuracies = np.ones(self.num_annotators)
        
    def vote(self, weighted=False, threshold=0.5): 
        
        self.majority = -1 * np.ones((self.num_words, 1))  
        votes = np.zeros((self.num_words, self.num_labels))     
        
        # iterate through words
        for i in xrange(self.num_words):
        
            # count all votes
            for j in xrange(self.num_annotators):
                votes[i, int(self.annotations[i,j])] += self.accuracies[j]
        
            # determine all labels with most votes
            winners = [(j,votes[i,j]) for j in xrange(self.num_labels) if votes[i,j] == votes[i,:].max()]
        
            # choose label
            if len(winners)==1:
                if winners[0][1] >= threshold:
                    self.majority[i] = winners[0][0]
                else:
                    self.majority[i] = 1
            else:
                if (not (1 in winners)) and (winners[0][1] >= threshold):
                    self.majority[i] = 2
                else:
                    self.majority[i] = 1
                    
        return self.majority, votes
                    
                    
    def update_accuracies(self):
        for i in xrange(self.num_annotators):
            self.accuracies[i] = metrics.accuracy_score(self.majority, self.annotations[:,i])
        
        return self.accuracies
        
        
    def run(self, weighted=False, threshold=0.5, max_iter=100):
    
        accuracies_old = np.ones(self.accuracies.shape) * -np.inf
        
        iteration = 0
    
        while (np.linalg.norm(self.accuracies-accuracies_old) >= 0.01) & (iteration < max_iter):
            iteration += 1
            
            self.vote(weighted, threshold)
            
            accuracies_old = self.accuracies
            self.update_accuracies()
        
        return self.majority