from data.data_generator import DataGenerator
from evaluation.experiment import Experiment

dataGen = DataGenerator('./config/data.ini', seed=42)

# Experiment(dataGen, './config/class_bias_experiment.ini').replot_results()
# Experiment(dataGen, './config/crowd_size_experiment.ini').replot_results()
Experiment(dataGen, './config/short_bias_experiment.ini').replot_results()
# Experiment(dataGen, './config/doc_length_experiment.ini').replot_results()
# Experiment(dataGen, './config/group_ratio_experiment.ini').replot_results()
# Experiment(dataGen, './config/acc_bias_experiment.ini').replot_results()