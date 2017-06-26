'''
Created on Oct 19, 2016

@author: Melvin Laux
'''
from src.data.data_generator import DataGenerator
from src.evaluation.experiment import Experiment

e = Experiment(DataGenerator('config/data.ini', seed=42), 'config/experiment.ini')

result = e.run_config()

if __name__ == '__main__':
    pass