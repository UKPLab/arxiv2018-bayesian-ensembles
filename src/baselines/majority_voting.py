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
        
    def vote(self, weighted=False, threshold=0.5, simple=True):
        '''
        If simple is True, we ignore the structure of the labels and just take the most popular
        :param weighted:
        :param threshold:
        :param simple:
        :return:
        '''
        votes = np.zeros((self.num_words, self.num_labels))

        for l in range(self.num_labels):
            votes[:, l] = np.sum((self.annotations==l).astype(int), axis=1)

        if simple: # we have a simple preference for other classes over 'O' if there is a tie-break situation

            # order used by Nugyen -- first one has preference in a tie.
            # missing -- 0
            # 'B-LOC': 1
            # 'B-MISC': 2
            # 'B-ORG': 3
            # 'B-PER': 4
            # 'I-LOC': 5
            # 'I-MISC': 6
            # 'I-ORG': 7
            # 'I-PER': 8
            # 'O': 9
            preferences = np.array([7, 9, 3, 8, 4, 5, 1, 6, 2])[:self.num_labels]
            # avoid choosing class 1 if there's a tie
            maxvotes = np.max(votes, axis=1)
            maxlabels = votes == maxvotes[:, None]

            self.majority = np.zeros(votes.shape[0], dtype=int)
            for i, row in enumerate(maxlabels):
                order = preferences[row]
                chosen = np.argwhere(preferences == np.min(order)).flatten()[0]
                self.majority[i] = chosen

            # remove the 1 scores temporarily if there is a tie. Keep the original votes for computing prob estimates.
            # ties_to_fix = (np.sum((votes == maxvotes[:, None]), axis=1) >= 2) & (votes[:, 1] == maxvotes)
            # votes_tmp = np.copy(votes)
            # votes_tmp[ties_to_fix, 1] = 0
            #
            # self.majority = np.argmax(votes_tmp, axis=1)

            # self.majority = np.argmax(votes, axis=1)
        else:
            self.majority = -np.ones((self.num_words, 1))

            I_labels = np.arange((self.num_labels - 1)/2) * 2 + 1
            I_labels[0] = 0
            I_labels = I_labels.astype(int)

            B_labels = np.arange((self.num_labels - 1)/2) * 2 + 2
            B_labels = B_labels.astype(int)

            # iterate through words
            for i in range(self.num_words):

                # count all votes
                # for j in range(self.num_annotators):
                #     if int(self.annos[i,j]) >= 0:
                #         votes[i, int(self.annos[i,j])] += self.accuracies[j]
                #
                # # determine all labels with most votes
                threshold_i = threshold * np.sum(votes[i, :])

                # choose label
                if votes[i, 1] > threshold_i:
                    self.majority[i] = 1 # decide if O or not
                else:
                    # choose between I or B. When tied, I is selected first.
                    I_or_B = np.argmax([np.sum(votes[i, I_labels]), np.sum(votes[i, B_labels])])

                    if I_or_B == 0: # I
                        # choose the class
                        vote_by_class = votes[I_labels, 0] + votes[B_labels, 0]
                        chosen_class = np.argmax(vote_by_class)
                        self.majority[i] = I_labels[chosen_class]
                    else: # B
                        # choose the class
                        vote_by_class = votes[I_labels, 0] + votes[B_labels, 0]
                        chosen_class = np.argmax(vote_by_class)
                        self.majority[i] = B_labels[chosen_class]

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