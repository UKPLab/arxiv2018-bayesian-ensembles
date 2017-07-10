'''
Created on Jul 6, 2017

@author: Melvin Laux
'''

import os
import pandas
import numpy as np

def convert(x):
    label = x.split('-')[0]
    if label=='I':
        return 0
    if label=='O':
        return 1
    if label=='B':
        return 2
    

path = '../../data/argmin/full0.dat.out'

data = pandas.read_csv(path,sep='\t', usecols=(0,5,6), converters={5:convert, 6:convert})

print data

print data.as_matrix()


if __name__ == '__main__':
    pass