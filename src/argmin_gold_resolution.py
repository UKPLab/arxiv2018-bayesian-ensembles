import pandas as pd
import numpy as np

# resolve the disagreements between three experts using a fourth expert to produce a gold evaluation set
from baselines.majority_voting import MajorityVoting

data_dir = '../../data/bayesian_sequence_combination/data/argmin_LMU/'

expert_file = data_dir + 'expert.csv'

gold_text = pd.read_csv(expert_file, sep=',', usecols=[5])
gold_doc_start = pd.read_csv(expert_file, sep=',', usecols=[1])
expert_labels = pd.read_csv(expert_file, sep=',', usecols=[2, 3, 4])

mv = MajorityVoting(expert_labels.values, 5)
gold, gold_probs = mv.run()

gold_output = pd.concat((gold_doc_start, expert_labels, pd.DataFrame(np.max(gold_probs, axis=1)), gold_text, pd.DataFrame(gold)), axis=1)
gold_output.to_csv(data_dir + 'expert_agreement.csv')