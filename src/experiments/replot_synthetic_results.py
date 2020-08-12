from data.data_generator import DataGenerator
from evaluation.synthetic_experiment import SynthExperiment

dataGen = DataGenerator('./config/data.ini', seed=42)

#Experiment(dataGen, config='./config/class_bias_experiment.ini').replot_results(exclude_methods=['bac_mace'], recompute=True)
#Experiment(dataGen, config='./config/acc_bias_experiment.ini').replot_results(exclude_methods=['bac_mace'], recompute=True)
#Experiment(dataGen, config='./config/short_bias_experiment.ini').replot_results(exclude_methods=['bac_mace'], recompute=True)
# Experiment(dataGen, config='./config/group_ratio_experiment.ini').replot_results(exclude_methods=['bac_mace'], recompute=True)

SynthExperiment(dataGen, config='./config/crowd_size_experiment.ini').replot_results()
# Experiment(dataGen, config='./config/doc_length_experiment.ini').replot_results()

