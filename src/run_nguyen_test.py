'''
Created on 4 Sep 2017

@author: simpson

A simple test to integrate the method available here:

https://github.com/thanhan/seqcrowd-acl17

From this paper:

Aggregating and Predicting Sequence Labels from Crowd Annotations
An T. Nguyen, Byron C. Wallace, Junyi Jessy Li, Ani Nenkova, Matthew Lease
ACL 2017


'''
from data.data_generator import DataGenerator
from evaluation.experiment import Experiment

if __name__ == '__main__':
    dataGen = DataGenerator('./config/data.ini', seed=42)
    Experiment(dataGen, './config/nguyen_class_bias_experiment.ini').run()
    