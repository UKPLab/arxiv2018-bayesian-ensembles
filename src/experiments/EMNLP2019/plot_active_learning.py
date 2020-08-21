import os

from evaluation.experiment import data_root_dir
from evaluation.plots import plot_active_learning_results

if __name__ == '__main__':
    print('Plotting active learning results...')

    results_dir = os.path.join(data_root_dir, 'output/ner_al/')
    output_dir = './documents/figures/NER_AL/'
    intervals = 'NER'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    plot_active_learning_results(results_dir, output_dir, intervals)