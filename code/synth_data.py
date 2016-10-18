'''
Created on Sep 11, 2016

@author: Melvin Laux
'''

from __future__ import division
import os
import numpy as np
import ConfigParser
from annotator import Annotator
from scipy.stats import dirichlet
from sys import argv
from sklearn.metrics import classification_report, accuracy_score, roc_curve
import matplotlib.pyplot as plt

schema = None

num_docs = None
doc_length = None
initial_probabilities = None
transition_probabilities = None

epsilon = 0.00000001

accuracy = None
miss_bias = None
long_bias = None

crowd_model = None
crowd_size = None

annotators = []

alphas = None

crowd_annotations = None 

data = None

output_dir = 'output/'

majority_threshold = None
worker_accuracies = None

priors = None
likelihood_gt = None
likelihood_prev = None

def simulate_ground_truth():
    global data
    
    data = np.ones((num_docs*doc_length, crowd_size + 2)) * -1
    
    for i in xrange(num_docs):
        
        # filling document index column
        data[i*doc_length:(i+1)*doc_length,0] = i
        
        doc = np.random.choice(range(len(schema)), 1, p = initial_probabilities)
        
        for j in xrange(doc_length - 1):
            #print doc
            doc = np.append(doc, np.random.choice(range(len(schema)), 1, p = transition_probabilities[doc[j], :]))
            
        data[i*doc_length:(i+1)*doc_length, 1] = np.transpose(doc)
    
    return


def simulate_crowd_annotators():
    global crowd_annotations
    global data
    
    for annotator in xrange(crowd_size):
        #seq = build_sequence(crowd_model[annotator,:,:,:], doc_length)
        #print seq
        for word in xrange(data.shape[0]):
            #print word, annotator
            true_label = data[word, 1]
            if (word < 1) or (data[word, 0] != data[word-1, 0]):
                previous_label = 3
            else:
                previous_label = data[word - 1, annotator + 2]
            data[word,annotator + 2] = annotators[annotator].annotate_word(previous_label, true_label)
            #data[word,annotator + 2] = np.asscalar(np.random.choice(range(len(schema)), 1, p = crowd_model[annotator, true_label, previous_label, :]))
            #crowd_annotations.append([np.asscalar(np.random.choice(range(len(schema)), 1, p = crowd_model[i, word, :])) for word in doc])
    #print data
    return

def build_sequence(model, length):
    
    sequence = []
    previous_label = 3

    for index in xrange(length):
        true_label = int(data[index, 1])
        #print model[true_label, previous_label, :]
        label = np.asscalar(np.random.choice(range(len(schema)), 1, p = model[true_label, previous_label, :]))
        #print sequence, int(true_label), previous_label, label
        sequence.append(label)
        previous_label = sequence[index]
    
    if is_valid(sequence):
        return sequence
    else:
        return build_sequence(model, length)


def is_valid(sequence):
    for i in xrange(len(sequence) - 1):    
        if (sequence[i]==1) & (sequence[i+1]==0):
            return False 
        
    return True


def build_crowd_model():
    global crowd_model
    global alphas
    
    alphas = np.ones((4,3,3)) 
    
    # add accuracy
    alphas += (np.identity(3)*accuracy)[None,:,:]
        
    # add miss bias
    alphas[:,:,1] += np.array([miss_bias,0,miss_bias])
    
    # add short bias
    alphas[:,0,2] += long_bias
    
    # set epsilons
    alphas[[1,3],:,0] = epsilon
    
    # build annotators    
    for i in xrange(crowd_size):    
        model = np.zeros(alphas.shape)
    
        for prev_label in xrange(4):
            for true_label in xrange(3):
                model[prev_label,true_label,:] = dirichlet(alphas[prev_label][true_label]).rvs()
            
        annotators.append(Annotator(model))
        

def read_config_file(file_path):
    parser = ConfigParser.ConfigParser()
    parser.read(file_path)
    
    for section in parser.sections():
        parameters = dict(parser.items(section))
        
        for p in parameters:
            parameters[p] = np.array(eval(parameters[p].split('#')[0].strip()))
                
        globals().update(parameters)
    
    return


def save_generated_data():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # TODO: check for existence, for now simply override
    np.savetxt(output_dir + 'data.csv', data, fmt='%s', header='generated data \nschema: ' + str(schema))
    #np.savetxt(output_dir + 'ground_truth.csv', ground_truth, fmt='%s', header='generated ground truth data \nschema: ' + str(schema))
    #np.savetxt(output_dir + 'crowd_annotations.csv', crowd_annotations, fmt='%s', header='generated crowd annotation data \nschema: '+ str(schema))
    return


def majority_voting(weighted = False):
    global data
    global worker_accuracies
    
    # initialise weights and stopping criterion
    worker_accuracies = np.ones((crowd_size, 1))
    worker_accuracies_old = worker_accuracies - 1
        
    votes = np.ones((num_docs*doc_length, 1))
    data = np.append(data, votes, 1)
    
    while np.linalg.norm(worker_accuracies-worker_accuracies_old) >= 0.01:
    
        #print np.linalg.norm(worker_accuracies-worker_accuracies_old)
        
        estimates = np.zeros((num_docs*doc_length ,len(schema)))
    
        # iterate through words
        for i in xrange(num_docs*doc_length):
        
            # initialise vote counts
            v = {label:0 for label in xrange(len(schema))}
            #estimates = np.zeros((num_docs*doc_length ,len(schema)))
        
            # count all votes
            for j in xrange(len(data[i,2:])-1):
                v[data[i,j+2]] += int(worker_accuracies[j])/sum(worker_accuracies)
                estimates[i,int(data[i,j+2])] += int(worker_accuracies[j])/sum(worker_accuracies)
                #print data[i,j+2], v
        
                # determine all labels with most votes
                winners = [(key,val) for key,val in v.items() if val == max(v.values())]
        
                # choose label
                if len(winners)==1:
                    if winners[0][1] >= majority_threshold:
                        votes[i] = winners[0][0]
                    else:
                            votes[i] = 1
                else:
                    if (not (1 in winners)) and (winners[0][1] >= majority_threshold):
                        votes[i] = 2
                    else:
                        votes[i] = 1
    
        worker_accuracies_old = worker_accuracies
        worker_accuracies = calculate_worker_accuracy(votes)
        
        for val in xrange(len(votes)):
            data[val,12] = votes[val] 
    
    plot_roc_curve(estimates)

    return

def plot_roc_curve(estimates):
    lines = []
    for i in xrange(len(schema)):
        fpr, tpr, thresholds = roc_curve(data[:,1], estimates[:,i], pos_label=i)
        lines.append(plt.plot(fpr,tpr,label='Line'))
    #plt.legend(handles=lines)
    plt.show() 
    return

def calculate_metrics():
    global data
    print classification_report(data[:,1],data[:,12], target_names=schema)
    return

def calculate_worker_accuracy(target):
    global worker_accuracies
    
    for annotator in xrange(crowd_size):
        worker_accuracies[annotator] = accuracy_score(target, data[:, annotator+2])
        
    return worker_accuracies

def generate_data(config_file):
    read_config_file(config_file)
    build_crowd_model()
    simulate_ground_truth()
    simulate_crowd_annotators()
    majority_voting()
    save_generated_data()
    calculate_metrics()

    
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        output_dir = argv[2]
    generate_data(sys.argv[1])