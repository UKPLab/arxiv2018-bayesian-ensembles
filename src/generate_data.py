'''
Created on Oct 25, 2016

@author: Melvin Laux
'''

from data.data_generator import DataGenerator
from baselines.majority_voting import MajorityVoting
from evaluation.experiment import Experiment
from baselines.clustering import Clustering
from data.load_data import load_crowdsourcing_data

#generator = DataGenerator('config/data.ini')

#gt, annos, _ = generator.generate_dataset(num_docs=2, doc_length=10, save_to_file=True)

#e = Experiment(generator, 'config/experiment.ini')

#result = e.run_config()

load_crowdsourcing_data()





if __name__ == '__main__':
    pass