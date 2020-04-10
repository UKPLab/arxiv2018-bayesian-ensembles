'''
Created on Oct 19, 2016

@author: Melvin Laux
'''
from data.data_generator import DataGenerator
from evaluation.synthetic_experiment import SynthExperiment



if __name__ == '__main__':

    #profile_out = 'output/profiler/stats'

    #if not os.path.exists(profile_out):
    #    os.makedirs(profile_out)

    dataGen = DataGenerator('./config/data.ini', seed=42)

    #SynthExperiment(dataGen, config='./config/class_bias_experiment.ini').run_synth()
    #SynthExperiment(dataGen, config='./config/short_bias_experiment.ini').run_synth()
    #SynthExperiment(dataGen, config='./config/group_ratio_experiment.ini').run_synth()
    #SynthExperiment(dataGen, config='./config/acc_bias_experiment.ini').run_synth()

    # Not used in paper:
    SynthExperiment(dataGen, config='./config/crowd_size_experiment.ini').run_synth()
    #SynthExperiment(dataGen, config='./config/doc_length_experiment.ini').run_synth()

    # For debugging:
    #SynthExperiment(dataGen, config='./config/debugging.ini').run_synth()