'''
Created on Oct 25, 2016

@author: Melvin Laux
'''

from data.data_generator import DataGenerator
import data.data_utils as dut
from baselines.majority_voting import MajorityVoting
from evaluation.experiment import Experiment
from baselines.clustering import Clustering

generator = DataGenerator('config/data.ini')

gt, annos, _ = generator.generate_dataset(num_docs=3, doc_length=3, save_to_file=True)

#e = Experiment(generator, 'config/experiment.ini')

#result = e.run_config()



if __name__ == '__main__':
    pass