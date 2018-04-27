'''
Created on Jul 6, 2017

@author: Melvin Laux
'''

import os
import pandas as pd
import numpy as np
import glob

def convert_argmin(x):
    label = x.split('-')[0]
    if label == 'I':
        return 0
    if label == 'O':
        return 1
    if label == 'B':
        return 2

def convert_7class_argmin(x):
    label = x.split('-')[0]
    if label == 'I':
        label = x.split('-')[1].split(':')[0]
        if label == 'MajorClaim':
            return 0
        elif label == 'Claim':
            return 3
        elif label == 'Premise':
            return 5
    if label == 'O':
        return 1
    if label == 'B':
        label = x.split('-')[1].split(':')[0]
        if label == 'MajorClaim':        
            return 2
        elif label == 'Claim':
            return 4
        elif label == 'Premise':
            return 6
    
def convert_crowdsourcing(x):
    if x == 'Premise-I':
        return 0
    elif x == 'O':
        return 1
    elif x== 'Premise-B':
        return 2
    else:
        return -1
    

def load_argmin_data():    

    path = '../../data/bayesian_annotator_combination/data/argmin/'
    if not os.path.isdir(path):
        os.mkdir(path)
            
    all_files = glob.glob(os.path.join(path, "*.dat.out"))
    df_from_each_file = (pd.read_csv(f, sep='\t', usecols=(0, 5, 6), converters={5:convert_argmin, 6:convert_argmin},
                                     header=None, quoting=3) for f in all_files)
    concatenated = pd.concat(df_from_each_file, ignore_index=True, axis=1).as_matrix()

    annos = concatenated[:, 1::3]
    
    for t in range(1, annos.shape[0]):
        annos[t, (annos[t-1, :] == 1) & (annos[t, :] == 0)] = 2
    
    gt = concatenated[:, 2][:, None]
    doc_start = np.zeros((annos.shape[0], 1))    
    doc_start[np.where(concatenated[:, 0] == 1)] = 1

    # correct the base classifiers
    non_start_labels = [0]
    start_labels = [2] # values to change invalid I tokens to
    for l, label in enumerate(non_start_labels):
        start_annos = annos[doc_start.astype(bool).flatten(), :]
        start_annos[start_annos == label] = start_labels[l]
        annos[doc_start.astype(bool).flatten(), :] = start_annos

    np.savetxt('../../data/bayesian_annotator_combination/data/argmin/annos.csv', annos, fmt='%s', delimiter=',')
    np.savetxt('../../data/bayesian_annotator_combination/data/argmin/gt.csv', gt, fmt='%s', delimiter=',')
    np.savetxt('../../data/bayesian_annotator_combination/data/argmin/doc_start.csv', doc_start, fmt='%s', delimiter=',')

    return gt, annos, doc_start

def load_argmin_7class_data():    

    path = '../../data/bayesian_annotator_combination/data/argmin/'
    if not os.path.isdir(path):
        os.mkdir(path)
    
    all_files = glob.glob(os.path.join(path, "*.dat.out"))
    df_from_each_file = (pd.read_csv(f, sep='\t', usecols=(0, 5, 6), converters={5:convert_7class_argmin,
                                                    6:convert_7class_argmin}, header=None, quoting=3) for f in all_files)
    concatenated = pd.concat(df_from_each_file, ignore_index=True, axis=1).as_matrix()

    annos = concatenated[:, 1::3]
    gt = concatenated[:, 2][:, None]
    doc_start = np.zeros((annos.shape[0], 1))    
    doc_start[np.where(concatenated[:, 0] == 1)] = 1

    # correct the base classifiers
    non_start_labels = [0, 3, 5]
    start_labels = [2, 4, 6] # values to change invalid I tokens to
    for l, label in enumerate(non_start_labels):
        start_annos = annos[doc_start.astype(bool).flatten(), :]
        start_annos[start_annos == label] = start_labels[l]
        annos[doc_start.astype(bool).flatten(), :] = start_annos

    outpath = '../../data/bayesian_annotator_combination/data/argmin7/'
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    
    np.savetxt(outpath + 'annos.csv', annos, fmt='%s', delimiter=',')
    np.savetxt(outpath + 'gt.csv', gt, fmt='%s', delimiter=',')
    np.savetxt(outpath + 'doc_start.csv', doc_start, fmt='%s', delimiter=',')

    return gt, annos, doc_start

def load_crowdsourcing_data():
    path = '../../data/bayesian_annotator_combination/data/crowdsourcing/'
    if not os.path.isdir(path):
        os.mkdir(path)
            
    all_files = glob.glob(os.path.join(path, "exported*.csv"))
    print(all_files)
    
    convs = {}
    for i in range(1,50):
        convs[i] = convert_crowdsourcing
    
    df_from_each_file = [pd.read_csv(f, sep=',', header=None, skiprows=1, converters=convs) for f in all_files]
    concatenated = pd.concat(df_from_each_file, ignore_index=False, axis=1).as_matrix()
    
    concatenated = np.delete(concatenated, 25, 1);
    
    annos = concatenated[:,1:]
    
    doc_start = np.zeros((annos.shape[0],1))
    doc_start[0] = 1    
    
    for i in range(1,annos.shape[0]):
        if '_00' in str(concatenated[i,0]):
            doc_start[i] = 1
    
    np.savetxt('../../data/bayesian_annotator_combination/data/crowdsourcing/gen/annos.csv', annos, fmt='%s', delimiter=',')
    np.savetxt('../../data/bayesian_annotator_combination/data/crowdsourcing/gen/doc_start.csv', doc_start, fmt='%s', delimiter=',')
    
    return annos, doc_start

if __name__ == '__main__':
    pass
