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

    def __init__(self, annotations, num_labels):
        '''
        Constructor
        '''
        
        self.annotations = annotations
        self.num_words, self.num_annotators = annotations.shape
        self.num_labels = num_labels
        self.accuracies = np.ones(self.num_annotators)
        self.probabilities = np.zeros((self.num_words, self.num_labels))
        
    def vote(self, weighted=False, threshold=0.5): 
        
        self.majority = -np.ones((self.num_words, 1))  
        votes = np.zeros((self.num_words, self.num_labels))     
        
        # iterate through words
        for i in range(self.num_words):
        
            # count all votes
            for j in range(self.num_annotators):
                if int(self.annotations[i,j]) >= 0:
                    votes[i, int(self.annotations[i,j])] += self.accuracies[j]
        
            # determine all labels with most votes
            winners = [(j,votes[i,j]) for j in range(self.num_labels) if votes[i,j] == votes[i,:].max()]
        
            # choose label
            if self.num_labels == 3:
                if len(winners)==1:
                    if winners[0][1] >= threshold:
                        self.majority[i] = winners[0][0]
                    else:
                        self.majority[i] = 1
                else:
                    if (not (1 in list(zip(*winners))[0])) and (winners[0][1] >= threshold):
                        self.majority[i] = 2
                    else:
                        self.majority[i] = 1
                        
            if self.num_labels == 7:
                # decide if 'O'
                if votes[i,1] >= np.sum(votes[i,[0,2,3,4,5,6]]):
                    self.majority[i] = 1
                else:
                    # decide between 'I' or 'B'
                    if np.sum(votes[i,[0,3,5]]) > np.sum(votes[i,[2,4,6]]):
                        # decide between classes
                        class_votes = np.array([np.sum(votes[i,[0,2]]),np.sum(votes[i,[3,4]]),np.sum(votes[i,[5,6]])])
                        if np.all(class_votes[0] >= class_votes):
                            self.majority[i] = 0
                        elif np.all(class_votes[1] >= class_votes):
                            self.majority[i] = 3
                        else:
                            self.majority[i] = 5
                    else:
                        # decide between classes
                        class_votes = np.array([np.sum(votes[i,[0,2]]),np.sum(votes[i,[3,4]]),np.sum(votes[i,[5,6]])])
                        if np.all(class_votes[0] >= class_votes):
                            self.majority[i] = 2
                        elif np.all(class_votes[1] >= class_votes):
                            self.majority[i] = 4
                        else:
                            self.majority[i] = 6
                
        
        if np.all(np.sum(votes, axis=1)[:,None]) != 0:
            self.probabilities = votes/np.sum(votes, axis=1)[:,None].astype(float)  

        return self.majority, self.probabilities
                    
                    
    def update_accuracies(self):
        for i in range(self.num_annotators):
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
        
        return self.majority, self.probabilities