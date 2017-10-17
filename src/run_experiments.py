'''
Created on Oct 19, 2016

@author: Melvin Laux
'''
from data.data_generator import DataGenerator
from evaluation.experiment import Experiment

#profile_out = 'output/profiler/stats'

#if not os.path.exists(profile_out):
#    os.makedirs(profile_out)

dataGen = DataGenerator('../config/data.ini', seed=42)
acc_exp = '../config/acc_bias_experiment.ini'



#Experiment(dataGen, '../config/class_bias_experiment.ini').run()
Experiment(dataGen, '../config/crowd_size_experiment.ini').run()
#Experiment(dataGen, '../config/short_bias_experiment.ini').run()
#Experiment(dataGen, '../config/doc_length_experiment.ini').run()
#Experiment(dataGen, '../config/group_ratio_experiment.ini').run()
Experiment(dataGen, acc_exp).run()


if __name__ == '__main__':
    pass