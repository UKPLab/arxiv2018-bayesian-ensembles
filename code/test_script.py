'''
Created on Oct 25, 2016

@author: Melvin Laux
'''
from __future__ import division
from data.data_generator import DataGenerator
import data.data_utils as dut
from baselines.majority_voting import MajorityVoting
from evaluation.experiment import Experiment
from baselines.clustering import Clustering


generator = DataGenerator('config/data.ini')

gt, annos, _ = generator.generate_dataset(save_to_file=True)

#print gt, annos

#c = Clustering(gt, annos)
#c.run_kde(1.0)
e = Experiment(generator, 'config/experiment.ini')

result = e.run_config()
#result = e.run(param_values, 10)


if __name__ == '__main__':
    pass