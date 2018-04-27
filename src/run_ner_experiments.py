'''
Created on April 27, 2018

@author: Edwin Simpson
'''

from evaluation.experiment import Experiment
import data.load_data as load_data
import numpy as np

outputdir = '../../data/bayesian_annotator_combination/output/ner/'

gt, annos, doc_start = load_data.load_ner_data() # TODO: write the data loader!

exp = Experiment(None, 3, annos.shape[1], None)
exp.methods = ['bac', 'ibcc', 'best', 'worst', 'majority']#['mace', 'best', 'worst', 'majority']#'bac', 'ibcc', 'mace', 'majority'] # 'bac', 'clustering', 'ibcc', 'mace',

exp.save_results = True
exp.opt_hyper = True

results, preds, _ = exp.run_methods(annos, gt, doc_start, -666, outputdir + 'annos.csv')

np.savetxt(outputdir + 'result_bacopt.csv', results, fmt='%s', delimiter=',', header=str(exp.methods).strip('[]'))
np.savetxt(outputdir + 'pred_bacopt.csv', preds, fmt='%s', delimiter=',', header=str(exp.methods).strip('[]'))

#results.dump('../../data/bayesian_annotator_combination/output/argmin/results')
#preds.dump('../../data/bayesian_annotator_combination/output/argmin/preds')

if __name__ == '__main__':
    pass
