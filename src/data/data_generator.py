'''
Created on Oct 18, 2016

@author: Melvin Laux
'''

import numpy as np
import os
import ConfigParser
import data_utils
from annotator import Annotator
from scipy.stats import dirichlet

class DataGenerator(object):
    '''
    classdocs
    '''
    gt_model = None
    crowd_model = None
    
    num_labels = None
    
    config_file = None
    
    output_dir = 'output/data/'
    

    def __init__(self, config_file):
        '''
        Constructor
        '''
        self.config_file = config_file 
        self.read_config_file()
        
        
    def read_config_file(self):
        parser = ConfigParser.ConfigParser()
        parser.read(self.config_file)
    
        # set up crowd model
        parameters = dict(parser.items('crowd_model'))
        self.init_crowd_model(float(parameters['acc_bias']),float(parameters['miss_bias']),float(parameters['short_bias']))
        
        # set up ground truth model
        parameters = dict(parser.items('ground_truth'))
        self.gt_model = np.array(eval(parameters['gt_model'].split('#')[0].strip()))
        
        # set up output destination
        parameters = dict(parser.items('output'))
        self.output_dir = parameters['output_dir'].split('#')[0].strip()
        
        # infer number of labels
        self.num_labels = self.gt_model.shape[1]
    
    
    def init_crowd_model(self, acc=None, miss=None, short=None):
        
        self.crowd_model = np.ones((4,3,3)) 
    
        # add accuracy
        self.crowd_model += (np.identity(3)*acc)[None,:,:]
        
        # add miss bias
        self.crowd_model[:,:,1] += np.array([miss,0,miss])
    
        # add short bias
        self.crowd_model[:,0,2] += short
    
        # set 'zero' values
        self.crowd_model[[1,3],:,0] = np.nextafter(0,1)
        
        
    def generate_ground_truth(self, num_docs, doc_length):
        '''
        Generates ground truth data using the transition and initial probabilities. The resulting data matrix has two 
        columns: column one contains the document indices and column two the label of a word. 
        '''
        
        data = np.ones((num_docs*doc_length, 2))*-1
        
        # generate documents
        for i in xrange(num_docs):
            # filling document index column
            data[i*doc_length:(i+1)*doc_length,0] = i
            
            # repeat creating documents...
            while True:
                data[i*doc_length, 1] = np.random.choice(range(self.num_labels), 1, p=self.gt_model[-1,:])
            
                # generate document content
                for j in xrange(i*doc_length + 1,(i+1)*doc_length):
                    data[j, 1] = np.random.choice(range(self.num_labels), 1, p=self.gt_model[int(data[j-1,1]), :])
                
                # ...break if document has valid syntax
                if data_utils.check_document_syntax(data[i*doc_length:(i+1)*doc_length,1]):
                    break

        return data
    
    
    def build_crowd(self, crowd_size):
        crowd = []
        
        # initialise annotators   
        while len(crowd) < crowd_size:    
            model = np.zeros(self.crowd_model.shape)
    
            for prev_label in xrange(self.num_labels+1):
                for true_label in xrange(self.num_labels):
                    model[prev_label,true_label,:] = dirichlet(self.crowd_model[prev_label][true_label]).rvs()
            
            crowd.append(Annotator(model))
            
        return crowd
    
    
    def generate_annotations(self, ground_truth, crowd):
        
        data = np.ones((ground_truth.shape[0], len(crowd)))
        
        # iterate through crowd
        for i in xrange(len(crowd)):    
            data[:,i] = crowd[i].annotate(ground_truth)
            
        return data

    
    def generate_dataset(self, num_docs=5, doc_length=10, crowd_size=10, save_to_file=False, output_dir=None):
        
        ground_truth = self.generate_ground_truth(num_docs, doc_length)
        crowd = self.build_crowd(crowd_size)
        
        annotations = self.generate_annotations(ground_truth, crowd)
        
        if save_to_file:
            if output_dir==None:
                output_dir = self.output_dir
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            np.savetxt(output_dir + 'full_data.csv', np.concatenate((ground_truth, annotations), 1), fmt='%s', delimiter=',')
            np.savetxt(output_dir + 'annotations.csv', annotations, fmt='%s', delimiter=',')
            np.savetxt(output_dir + 'ground_truth.csv', ground_truth, fmt='%s', delimiter=',')
        
        return ground_truth, annotations, crowd


if __name__ == '__main__':
    import sys
    generator = DataGenerator(sys.argv[1])
    generator.generate_dataset()