'''
Created on Oct 25, 2016

@author: Melvin Laux
'''

import sklearn.metrics as skm
import matplotlib.pyplot as plt
import numpy as np
    
def plot_roc_curve(ground_truth, votes):
    
    schema = ['I', 'O', 'B']

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in xrange(3):
        fpr, tpr, _ = skm.roc_curve(ground_truth[:,1], votes[:,i], pos_label=i)
        ax.plot(fpr, tpr, label='$%s\ vs\ all$' % schema[i], lw=2)

    ax.legend(loc='lower right')
    plt.title('ROC curves per label')
    plt.ylabel('true positive rate')
    plt.xlabel('false positive rate')

    plt.show()
    
def plot_metrics(ground_truth, majority):
    p, r, f1, s = skm.precision_recall_fscore_support(ground_truth[:,1], majority, average='micro')
    print p,r,f1,s