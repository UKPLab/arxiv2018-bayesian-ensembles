'''
Created on Jul 11, 2017

@author: Melvin Laux
'''

import numpy as np

'''
Calculates the absolute difference of numbers of found annotations between the prediction and the target sequence. 
This is done by counting the number of 'B' tokens in each sequence. 
'''        
def count_error(pred, target):
    return len(pred==2) - len(target==2)

'''
Calculates the number of invalid labels in the given prediction, i.e. the number of 'I' tokens directly following an 'O' token.
'''        
def num_invalid_labels(pred, doc_start, restricted=[0, 3, 5, 7], outside=1):

    docs = np.split(pred, np.where(doc_start==1)[0])[1:]
    try:
        count_invalid = lambda doc: len(np.where(np.in1d(doc[np.r_[0, np.where(doc[:-1] == outside)[0] + 1]], restricted))[0])
    except:
        print('Could not count any invalid labels because some classes were not present.')
        count_invalid = 0
    return np.sum(list(map(count_invalid, docs))) #/float(len(pred))

'''
Calculates the difference of the mean length of annotations between prediction and target sequences.
'''        
def mean_length_error(pred, target, doc_start):
    
    get_annos = lambda x: np.split(x, np.union1d(np.where(doc_start==1)[0], np.where(x==2)[0]))
    
    del_o = lambda x: np.delete(x, np.where(x==1)[0])
    
    not_empty = lambda x: np.size(x) > 0
    
    mean_pred = np.mean(np.array([len(x) for x in filter(not_empty, list(map(del_o, get_annos(pred))))]))
    if mean_pred == np.nan:
        mean_pred = 0
    mean_target = np.mean(np.array([len(x) for x in filter(not_empty, list(map(del_o, get_annos(target))))]))

    return np.abs(mean_pred-mean_target)

if __name__ == '__main__':
    pass