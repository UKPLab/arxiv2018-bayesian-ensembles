'''
Created on Jul 6, 2017

@author: Melvin Laux
'''

import os
import pandas as pd
import numpy as np
import glob

def convert(x):
    label = x.split('-')[0]
    if label=='I':
        return 0
    if label=='O':
        return 1
    if label=='B':
        return 2
    

def load_argmin_data():    

    path = '../../data/argmin/'
    all_files = glob.glob(os.path.join(path, "*.dat.out"))
    df_from_each_file = (pd.read_csv(f, sep='\t',  usecols=(0,5,6), converters={5:convert,6:convert}, header=None) for f in all_files)
    concatenated = pd.concat(df_from_each_file, ignore_index=True, axis=1).as_matrix()

    annos = concatenated[:,1::3]
    gt = concatenated[:,2][:,None]
    doc_start = np.zeros((annos.shape[0],1))    
    doc_start[np.where(annos[:,0]==1)] = 1

    return gt, annos, doc_start


if __name__ == '__main__':
    pass