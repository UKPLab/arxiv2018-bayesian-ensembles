import os
from evaluation.plots import plot_active_learning_results

if __name__ == '__main__':
    print('Plotting active learning results...')

    # results_dir = '../../data/bayesian_annotator_combination/output/ner_al_new2/'
    # output_dir = './documents/figures/NER_AL/'
    # intervals = 'NER'

    # results_dir = '../../data/bayesian_annotator_combination/output/ner_rand_new2/'
    # output_dir = './documents/figures/NER_RAND/'
    # intervals = 'NER'

    # results_dir = '../../data/bayesian_annotator_combination/output/bio_al_new/'
    # output_dir = './documents/figures/PICO_AL/'
    # intervals = 'PICO'

    results_dir = '../../data/bayesian_annotator_combination/output/bio_rand_new/'
    output_dir = './documents/figures/PICO_RAND/'
    intervals = 'PICO'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    plot_active_learning_results(results_dir, output_dir, intervals)#, result_str='recomputed_')