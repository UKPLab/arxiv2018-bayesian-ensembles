'''
SYNTHETIC DATA EXPERIMENTS

Generate the data and run the experiments.

'''
import configparser
import os
import numpy as np

from evaluation.metrics import calculate_scores
from evaluation.plots import SCORE_NAMES, plot_results
from evaluation.experiment import data_root_dir, Experiment
import glob
import pandas as pd


class SynthExperiment():

    def __init__(self, generator, nclasses=None, nannotators=None, config=None,
                 alpha0_factor=1.0, alpha0_diags=1.0,
                 beta0_factor=0.1, max_iter=20, rep=0,
                 max_internal_iter=3, begin_factor=1.0):

        self.output_dir = os.path.join(data_root_dir, 'output/')

        self.generator = generator

        self.num_runs = None
        self.doc_length = None
        self.group_sizes = None
        self.num_docs = None

        self.acc_bias = None
        self.miss_bias = None
        self.short_bias = None

        self.config_file = None

        self.param_values = None
        self.param_idx = None
        self.param_fixed = None

        self.generate_data = False

        self.save_results = True
        self.save_plots = False
        self.show_plots = False

        if config is not None:
            self.config_file = config
            self._read_config_file()

        else:
            self.num_classes = nclasses
            self.nannotators = nannotators

        self.alpha0_factor = alpha0_factor
        self.alpha0_diags = alpha0_diags
        self.begin_factor = begin_factor
        self.beta0_factor = beta0_factor
        self.max_internal_iter = max_internal_iter
        self.crf_probs = False

        # save results from methods here. If we use compound methods, we can reuse these results in different
        # combinations of methods.
        self.aggs = {}
        self.probs = {}

        self.max_iter = max_iter # allow all methods to use a maximum no. iterations

        np.random.seed(3849)
        self.seed = np.random.randint(1, 1000, 100)[rep] # seeds for AL

    def _read_config_file(self):

        print('Reading experiment config file...')

        parser = configparser.ConfigParser()
        parser.read(self.config_file)

        # set up parameters
        parameters = dict(parser.items('parameters'))
        self.param_idx = int(parameters['idx'].split('#')[0].strip())
        self.param_values = np.array(eval(parameters['values'].split('#')[0].strip()))

        self.acc_bias = np.array(eval(parameters['acc_bias'].split('#')[0].strip()))
        self.miss_bias = np.array(eval(parameters['miss_bias'].split('#')[0].strip()))
        self.short_bias = np.array(eval(parameters['short_bias'].split('#')[0].strip()))


        self.methods = eval(parameters['methods'].split('#')[0].strip())

        self.postprocess = eval(parameters['postprocess'].split('#')[0].strip())

        print(self.methods)

        self.num_runs = int(parameters['num_runs'].split('#')[0].strip())
        self.doc_length = int(parameters['doc_length'].split('#')[0].strip())
        self.group_sizes = np.array(eval(parameters['group_sizes'].split('#')[0].strip()))
        self.num_classes = int(parameters['num_classes'].split('#')[0].strip())
        self.num_docs = int(parameters['num_docs'].split('#')[0].strip())
        self.generate_data = eval(parameters['generate_data'].split('#')[0].strip())

        # set up output
        parameters = dict(parser.items('output'))
        self.output_dir = os.path.join(data_root_dir, parameters['output_dir'])
        print('Output directory: %s' % self.output_dir)
        self.save_results = eval(parameters['save_results'].split('#')[0].strip())
        self.save_plots = eval(parameters['save_plots'].split('#')[0].strip())
        self.show_plots = eval(parameters['show_plots'].split('#')[0].strip())


    def _create_experiment_data(self):
        for param_idx in range(self.param_values.shape[0]):
            # update tested parameter
            if self.param_idx == 0:
                self.acc_bias = self.param_values[param_idx]
            elif self.param_idx == 1:
                self.miss_bias = self.param_values[param_idx]
            elif self.param_idx == 2:
                self.short_bias = self.param_values[param_idx]
            elif self.param_idx == 3:
                self.num_docs = self.param_values[param_idx]
            elif self.param_idx == 4:
                self.doc_length = self.param_values[param_idx]
            elif self.param_idx == 5:
                self.group_sizes = self.param_values[param_idx]
            else:
                print('Encountered invalid test index!')

            param_path = self.output_dir + 'data/param' + str(param_idx) + '/'

            for i in range(self.num_runs):
                path = param_path + 'set' + str(i) + '/'
                self.generator.init_crowd_models(self.acc_bias, self.miss_bias, self.short_bias, self.group_sizes)
                self.generator.generate_dataset(num_docs=self.num_docs, doc_length=self.doc_length,
                                                group_sizes=self.group_sizes, save_to_file=True, output_dir=path)


    def _run_synth_exp(self):
        # initialise result array
        results = np.zeros((self.param_values.shape[0], len(SCORE_NAMES), len(self.methods), self.num_runs))

        # iterate through parameter settings
        for param_idx in range(self.param_values.shape[0]):

            print('parameter setting: {0}'.format(param_idx))

            # read parameter directory
            path_pattern = self.output_dir + 'data/param{0}/set{1}/'

            # iterate through data sets
            for run_idx in range(self.num_runs):
                print('data set number: {0}'.format(run_idx))

                data_path = path_pattern.format(*(param_idx, run_idx))

                if os.path.exists(os.path.join(data_path, 'pred_.csv')):
                    os.remove(os.path.join(data_path, 'pred_.csv'))
                if os.path.exists(os.path.join(data_path, 'probs_.csv')):
                    os.remove(os.path.join(data_path, 'probs_.csv'))

                # read data
                doc_start, gt, annos = self.generator.read_data_file(data_path + 'full_data.csv')
                exp = Experiment(data_path, self.generator.num_labels, annos, gt, doc_start, None,
                                 alpha0_factor=self.alpha0_factor, alpha0_diags=self.alpha0_diags,
                                 beta0_factor=self.beta0_factor, begin_factor=self.begin_factor,
                                 max_iter=self.max_iter, crf_probs=self.crf_probs, bootstrapping=False)
                exp.methods = self.methods
                # run methods
                results[param_idx, :, :, run_idx], preds, probabilities, _, _, _ = exp.run_methods(new_data=True,
                                                                    save_with_timestamp=False)


        if self.save_results:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            print('Saving results...')
            # np.savetxt(self.output_dir + 'results.csv', np.mean(results, 3)[:, :, 0])
            results.dump(self.output_dir + 'results')

        if self.show_plots or self.save_plots:
            plot_results(self.param_values, self.methods, self.param_idx, results, self.show_plots,
                         self.save_plots, self.output_dir)

        return results


    def replot_results(self, exclude_methods=[], recompute=True):
        # Reloads predictions and probabilities from file, then computes metrics and plots.

        # initialise result array
        results = np.zeros((self.param_values.shape[0], len(SCORE_NAMES), len(self.methods), self.num_runs))
        std_results = np.zeros((self.param_values.shape[0], len(SCORE_NAMES) - 3, len(self.methods), self.num_runs))

        if recompute:

            # iterate through parameter settings
            for param_idx in range(self.param_values.shape[0]):

                print('parameter setting: {0}'.format(param_idx))

                # read parameter directory
                path_pattern = self.output_dir + 'data/param{0}/set{1}/'

                # iterate through data sets
                for run_idx in range(self.num_runs):
                    print('data set number: {0}'.format(run_idx))

                    data_path = path_pattern.format(*(param_idx, run_idx))
                    # read data
                    doc_start, gt, annos = self.generator.read_data_file(data_path + 'full_data.csv')
                    # run methods
                    preds = pd.read_csv(data_path + 'pred_.csv').values
                    if preds.ndim == 1:
                        preds = preds[:, None]

                    print(data_path + 'probabilities')
                    probabilities = np.load(data_path + 'probs_.pkl', allow_pickle=True)
                    if probabilities.ndim == 2:
                        probabilities = probabilities[:, :, None]

                    for method_idx in range(len(self.methods)):

                        if self.methods[method_idx] in exclude_methods:
                            continue

                        print('running method: {0}'.format(self.methods[method_idx]), end=' ')

                        agg = preds[:, method_idx]
                        probs = probabilities[:, :, method_idx]

                        results[param_idx, :, method_idx, run_idx][:, None], \
                        std_results[param_idx, :, method_idx, run_idx] = calculate_scores(
                            self.postprocess, agg, gt.flatten(), probs, doc_start,
                            bootstrapping=False, print_per_class_results=True, )

        else:
            print('Loading precomputed results...')
            # np.savetxt(self.output_dir + 'results.csv', np.mean(results, 3)[:, :, 0])
            results = np.load(self.output_dir + 'results')

        included_methods = np.ones(len(self.methods), dtype=bool)
        included_methods[np.in1d(self.methods, exclude_methods)] = False
        included_methods = np.argwhere(included_methods).flatten()

        results = results[:, :, included_methods, :]

        methods = np.array(self.methods)[included_methods]

        if self.save_results:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            print('Saving results...')
            # np.savetxt(self.output_dir + 'results.csv', np.mean(results, 3)[:, :, 0])
            results.dump(self.output_dir + 'results')

        if self.show_plots or self.save_plots:
            print('making plots for parameter setting: {0}'.format(self.param_idx))
            plot_results(self.param_values, methods, self.param_idx, results,
                         self.show_plots, self.save_plots, self.output_dir)

        return results


    def run_synth(self):
        if self.generate_data:
            self._create_experiment_data()

        if not glob.glob(os.path.join(self.output_dir, 'data/*')):
            print('No data found at specified path! Exiting!')
            return
        else:
            return self._run_synth_exp()