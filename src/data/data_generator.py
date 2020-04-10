'''

Simulated data generation.

Created on Oct 18, 2016

@author: Melvin Laux
'''

import numpy as np
import os
import configparser
import data.data_utils as data_utils
from data.annotator import Annotator
from scipy.stats import dirichlet

class DataGenerator(object):
    '''
    classdocs
    '''
    gt_model = None
    crowd_model = None
    
    num_labels = None
    group_sizes = None
    
    config_file = None
    
    output_dir = 'output/data/'
    

    def __init__(self, config_file, seed=None):
        '''
        Constructor
        '''
        self.config_file = config_file 
        self.read_config_file()
        if not seed==None:
            np.random.seed(seed)
            
    
    def read_data_file(self, path):
        data = np.genfromtxt(path, delimiter=',')
        doc_start = data[:,0]
        gt = data[:,1]
        annos = data[:,2:]
        
        return doc_start[:,None], gt[:,None], annos
        
    def read_config_file(self):
        parser = configparser.ConfigParser()
        parser.read(self.config_file)
        
        # set up crowd model
        parameters = dict(parser.items('crowd_model'))
        group_sizes = np.array(eval(parameters['group_sizes'].split('#')[0].strip()))
        self.init_crowd_models(np.array(eval(parameters['acc_bias'])),np.array(eval(parameters['miss_bias'])),np.array(eval(parameters['short_bias'])), group_sizes)
        #self.group_sizes = np.array(eval(parameters['group_sizes']))
        
        # set up ground truth model
        parameters = dict(parser.items('ground_truth'))
        self.gt_model = np.array(eval(parameters['gt_model'].split('#')[0].strip()))
        
        # set up output destination
        parameters = dict(parser.items('output'))
        self.output_dir = parameters['output_dir'].split('#')[0].strip()
        
        # infer number of labels
        self.num_labels = self.gt_model.shape[1]
    
    
    def init_crowd_models(self, acc=None, miss=None, short=None, group_sizes=None):

        if np.isscalar(acc):
            acc = [acc]
        if np.isscalar(miss):
            miss = [miss]
        if np.isscalar(short):
            short = [short]

        acc = np.array(acc)
        miss = np.array(miss)
        short = np.array(short)
        self.crowd_model = np.ones((4,3,3, acc.size))

        for i in range(acc.size):
            # add accuracy
            self.crowd_model[:,:,:,i] += (np.eye(3)*acc[i])[None,:,:]
        
            # add miss bias
            self.crowd_model[:,:,1,i] += np.array([miss[i],0,miss[i]])
    
            # add short bias
            self.crowd_model[:,0,2,i] += short[i]
    
        # set 'zero' values
        self.crowd_model[[1,3],:,0,:] = np.nextafter(0,1)
        
        
    def generate_ground_truth(self, num_docs, doc_length):
        '''
        Generates ground truth data using the transition and initial probabilities. The resulting data matrix has two 
        columns: column one contains the document indices and column two the label of a word. 
        '''
        
        doc_length = int(doc_length)
        
        data = -np.ones((num_docs*doc_length, 1))
        
        # generate documents
        for i in range(num_docs):
            # filling document index column
            #data[i*doc_length:(i+1)*doc_length,0] = i
            
            # repeat creating documents...
            while True:
                data[i*doc_length] = np.random.choice(list(range(self.num_labels)), 1, p=self.gt_model[-1,:])
            
                # generate document content
                for j in range(i*doc_length + 1,(i+1)*doc_length):
                    data[j] = np.random.choice(list(range(self.num_labels)), 1, p=self.gt_model[int(data[j-1]), :])
                
                # ...break if document has valid syntax
                if data_utils.check_document_syntax(data[i*doc_length:(i+1)*doc_length]):
                    break

        return data
    
    
    def build_crowd(self, group_sizes):
        crowd = []

        if np.isscalar(group_sizes):
            group_sizes = [group_sizes]

        group_sizes = np.array(group_sizes)
        
        # initialise annotators   
        for i in range(len(group_sizes)):    
            for j in range(group_sizes[i]):
                model = np.zeros(self.crowd_model[:,:,:,0].shape)
    
                for prev_label in range(self.num_labels+1):
                    for true_label in range(self.num_labels):
                        model[prev_label,true_label,:] = dirichlet(self.crowd_model[prev_label,true_label,:,i]).rvs()
            
                crowd.append(Annotator(model))
            
        return crowd
    
    
    def generate_annotations(self, doc_start, ground_truth, crowd):
        
        data = np.ones((ground_truth.shape[0], len(crowd)))
        
        # iterate through crowd
        for i in range(len(crowd)):    
            data[:,i] = crowd[i].annotate(ground_truth, doc_start)
            
        return data

    
    def generate_dataset(self, num_docs=5, doc_length=10, group_sizes=10, save_to_file=False, output_dir=None):
        
        ground_truth = self.generate_ground_truth(num_docs, doc_length)
        
        doc_start = np.zeros_like(ground_truth)
        doc_start[np.arange(num_docs)*doc_length] = 1
        
        crowd = self.build_crowd(group_sizes)
        
        annotations = self.generate_annotations(doc_start, ground_truth, crowd)

        doc_start = np.zeros((num_docs * int(doc_length), 1))
        doc_start[np.array(range(0, int(num_docs * doc_length), doc_length))] = 1

        if save_to_file:
            if output_dir==None:
                output_dir = self.output_dir
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            np.savetxt(output_dir + 'full_data.csv', np.concatenate((doc_start,ground_truth, annotations),1), fmt='%s', delimiter=',')
            np.savetxt(output_dir + 'annos.csv', annotations, fmt='%s', delimiter=',')
            np.savetxt(output_dir + 'ground_truth.csv', ground_truth, fmt='%s', delimiter=',')
            np.savetxt(output_dir + 'doc_start.csv', doc_start, fmt='%s', delimiter=',')
        
        return ground_truth, annotations, doc_start


if __name__ == '__main__':
    import sys
    generator = DataGenerator(sys.argv[1])
    generator.generate_dataset()
