'''
Created on Nov 1, 2016

@author: Melvin Laux
'''

from baselines import ibcc, clustering, majority_voting
from algorithm import bac
from data import data_utils
import ConfigParser
import os, subprocess
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt
import metrics
import glob
from src.baselines.hmm import HMM_crowd
from src.baselines.util import crowd_data
from baselines.util import crowdlab, instance

class Experiment(object):
    '''
    classdocs
    '''
    
    generator = None
    
    config_file = None
    
    param_values = None
    param_idx = None
    param_fixed = None
    
    acc_bias = None
    miss_bias = None
    short_bias = None
    
    PARAM_NAMES = ['acc_bias', 'miss_bias', 'short_bias', 'num_docs', 'doc_length', 'group_sizes']
    SCORE_NAMES = ['accuracy', 'precision', 'recall', 'f1-score', 'auc-score', 'cross-entropy-error', 'count error', 'number of invalid labels', 'mean length error']
    
    generate_data= False
    
    methods = None
    
    num_runs = None
    doc_length = None
    group_sizes = None
    num_docs = None
    
    num_classes = None
    
    output_dir = '/output/'
    
    save_results = False
    save_plots = False
    show_plots = False
    
    postprocess = True
    
    '''
    Constructor
    '''
    def __init__(self, generator, config=None):
        '''
        Constructor
        '''
        self.generator = generator
        
        if not (config == None):
            self.config_file = config
            self.read_config_file()
            
            
    def read_config_file(self):
        
        print 'Reading experiment config file...'
        
        parser = ConfigParser.ConfigParser()
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
        
        print self.methods
                
        self.num_runs = int(parameters['num_runs'].split('#')[0].strip())
        self.doc_length = int(parameters['doc_length'].split('#')[0].strip())
        self.group_sizes = np.array(eval(parameters['group_sizes'].split('#')[0].strip()))
        self.num_classes = int(parameters['num_classes'].split('#')[0].strip())
        self.num_docs = int(parameters['num_docs'].split('#')[0].strip())
        self.generate_data = eval(parameters['generate_data'].split('#')[0].strip())
        
        # set up output
        parameters = dict(parser.items('output'))
        self.output_dir = parameters['output_dir']
        self.save_results = eval(parameters['save_results'].split('#')[0].strip())
        self.save_plots = eval(parameters['save_plots'].split('#')[0].strip())
        self.show_plots = eval(parameters['show_plots'].split('#')[0].strip())
        
    
    def run_methods(self, annotations, ground_truth, doc_start, param_idx, anno_path):
        
        scores = np.zeros((len(self.SCORE_NAMES), len(self.methods)))
        predictions = -np.ones((annotations.shape[0], len(self.methods)))
        
        for method_idx in xrange(len(self.methods)): 
            
            print 'running method: {0}'.format(self.methods[method_idx]), 
            
            if self.methods[method_idx] == 'majority':
                    
                agg, probs = majority_voting.MajorityVoting(annotations, self.num_classes).vote()
            
            if self.methods[method_idx] == 'clustering':
                    
                cl = clustering.Clustering(ground_truth, annotations, doc_start)
                agg = cl.run()
                    
                probs = np.zeros((ground_truth.shape[0], self.num_classes))
                for k in xrange(ground_truth.shape[0]):
                    probs[k, int(agg[k])] = 1      
                
            if self.methods[method_idx] == 'mace':
                
                devnull = open(os.devnull, 'w')
                subprocess.call(['java', '-jar', '../MACE/MACE.jar', '--distribution', '--prefix', '../output/data/mace', anno_path], stdout = devnull, stderr = devnull)
                
                result = np.genfromtxt('../output/data/mace.prediction')
                    
                agg = result[:, 0]
                    
                probs = np.zeros((ground_truth.shape[0], self.num_classes))
                for i in xrange(result.shape[0]):
                    for j in xrange(0, self.num_classes * 2, 2):
                        probs[i, int(result[i, j])] = result[i, j + 1]  
                
            if self.methods[method_idx] == 'ibcc':
                
                alpha0 = np.ones((self.num_classes, self.num_classes, 10))
                alpha0[:, :, 5] = 2.0
                alpha0[np.arange(3), np.arange(3), :] += 1.0
                nu0 = np.array([1, 1, 1], dtype=float)
                
                ibc = ibcc.IBCC(nclasses=3, nscores=3, nu0=nu0, alpha0=alpha0)
                probs = ibc.combine_classifications(annotations, table_format=True)
                agg = probs.argmax(axis=1)
                
            if self.methods[method_idx] == 'bac':
                
                alg = bac.BAC(L=3, K=annotations.shape[1])
                probs = alg.run(annotations, doc_start)
                agg = probs.argmax(axis=1)
                       
            if self.methods[method_idx] == 'HMM_crowd':
                sentences = []
                crowd_labels = []
                for i in range(annotations.shape[0]):
                    if doc_start[i]:
                        sentence = []
                        sentence.append(instance([], 0))
                        sentences.append(sentence)
                        crowd_labs = []
                        for worker in range(annotations.shape[1]):
                            worker_labs = crowdlab(worker, len(sentences) - 1, [int(annotations[i, worker])])
                
                            crowd_labs.append(worker_labs)
                        crowd_labels.append(crowd_labs)
                        
                    else:
                        sentence.append(instance([], 0))
                        for worker in range(annotations.shape[1]):
                            crowd_labs[worker].sen.append(int(annotations[i,worker]))
                
                data = crowd_data(sentences, crowd_labels)
                hc = HMM_crowd(3, 1, data, None, None, vb=[0.1, 0.1], ne=1)
                hc.init(init_type = 'dw', wm_rep='cv2', dw_em=5, wm_smooth=0.1)
                hc.em(20)
                hc.mls()              
                agg = np.array(hc.res).flatten() 
                probs = []
                for sentence_post_arr in hc.sen_posterior:
                    for tok_post_arr in sentence_post_arr:
                        probs.append(tok_post_arr)
                probs = np.array(probs)
            
            scores[:,method_idx][:,None] = self.calculate_scores(agg, ground_truth, probs, doc_start)
            predictions[:,method_idx] = agg.flatten()            
            
            print '...done'
            
        return scores, predictions
    
    
    def calculate_scores(self, agg, gt, probs, doc_start):
        
        result = -np.ones((9,1))
        
        result[7] = metrics.num_invalid_labels(agg, doc_start)
        
        if self.postprocess:
            agg = data_utils.postprocess(agg, doc_start)
        
        result[0] = skm.accuracy_score(gt, agg)
        result[1] = skm.precision_score(gt, agg, average='macro')
        result[2] = skm.recall_score(gt, agg, average='macro')
        result[3] = skm.f1_score(gt, agg, average='macro')
        auc_score = skm.roc_auc_score(gt==0, agg==0) * np.sum(gt==0)
        auc_score += skm.roc_auc_score(gt==1, agg==1) * np.sum(gt==1)
        auc_score += skm.roc_auc_score(gt==2, agg==2) * np.sum(gt==2)
        result[4] = auc_score / float(gt.shape[0])
        result[5] = skm.log_loss(gt, probs, eps=1e-100)
        result[6] = metrics.abs_count_error(agg, gt)
        result[8] = metrics.mean_length_error(agg, gt, doc_start)
        
        return result
        
    def create_experiment_data(self):
        for param_idx in xrange(self.param_values.shape[0]):
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
                print 'Encountered invalid test index!'
                
            param_path = self.output_dir + 'data/param' + str(param_idx) + '/'
        
            for i in xrange(self.num_runs):
                path = param_path + 'set' + str(i) + '/'
                self.generator.init_crowd_models(self.acc_bias, self.miss_bias, self.short_bias, self.group_sizes)
                self.generator.generate_dataset(num_docs=self.num_docs, doc_length=self.doc_length, group_sizes=self.group_sizes, save_to_file=True, output_dir=path)
    
    def run_exp(self):
        # initialise result array
        results = np.zeros((self.param_values.shape[0], len(self.SCORE_NAMES), len(self.methods), self.num_runs))
        # read experiment directory
        param_dirs = glob.glob(os.path.join(self.output_dir, "data/*"))
       
        # iterate through parameter settings
        for param_idx in xrange(len(param_dirs)):
            
            print 'parameter setting: {0}'.format(param_idx)
            
            # read parameter directory 
            path_pattern = self.output_dir + 'data/param{0}/set{1}/'

            # iterate through data sets
            for run_idx in xrange(self.num_runs):
                print 'data set number: {0}'.format(run_idx)
                
                data_path = path_pattern.format(*(param_idx,run_idx))
                # read data
                doc_start, gt, annos = self.generator.read_data_file(data_path + "full_data.csv")                
                # run methods
                results[param_idx,:,:,run_idx], preds = self.run_methods(annos, gt, doc_start, param_idx, data_path + "annotations.csv")
                # save predictions
                np.savetxt(data_path + "predictions.csv", preds)
                
        if self.save_results:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
            print 'Saving results...'
            #np.savetxt(self.output_dir + 'results.csv', np.mean(results, 3)[:, :, 0])
            results.dump(self.output_dir + 'results')

        if self.show_plots or self.save_plots:
            self.plot_results(results, self.show_plots, self.save_plots, self.output_dir)
        
        return results
    
    def run(self):
        
        if self.generate_data:
            self.create_experiment_data()
        
        if not glob.glob(os.path.join(self.output_dir, "data/*")):
            print 'No data found at specified path! Exiting!'
            return
        else:
            return self.run_exp()
        
        
    def make_plot(self, x_vals, y_vals, x_ticks_labels, ylabel):
        for j in xrange(len(self.methods)):
            plt.errorbar(x_vals, np.mean(y_vals[:, j, :], 1), yerr=np.std(y_vals[:, j, :], 1), label=self.methods[j])
                  
        plt.legend(loc='best')
        
        plt.title('parameter influence')
        plt.ylabel(ylabel)
        plt.xlabel(self.PARAM_NAMES[self.param_idx])
        plt.xticks(x_vals, x_ticks_labels)
        plt.ylim([0, np.max([1, np.max(y_vals)])])
    
    
    def plot_results(self, results, show_plot=False, save_plot=False, output_dir='/output/'):
        
        # create output directory if necessary
        if save_plot and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # initialise values for x axis
        if (self.param_values.ndim > 1):
            x_vals = self.param_values[:, 0]
        else:
            x_vals = self.param_values
            
        # initialise x-tick labels 
        x_ticks_labels = map(str, self.param_values)
            
        for i in xrange(len(self.SCORE_NAMES)):  
            self.make_plot(x_vals, results[:,i,:,:], x_ticks_labels, self.SCORE_NAMES[i])
        
            if save_plot:
                print 'Saving plot...'
                plt.savefig(self.output_dir + 'plot_' + self.SCORE_NAMES[i] + '.png') 
                plt.clf()
        
            if show_plot:
                plt.show()
                
            if i == 5:
                self.make_plot(x_vals, results[:,i,:,:], x_ticks_labels, self.SCORE_NAMES[i])
                plt.ylim([0,1])
                
                if save_plot:
                    print 'Saving plot...'
                    plt.savefig(self.output_dir + 'plot_' + self.SCORE_NAMES[i] + '_zoomed' + '.png') 
                    plt.clf()
        
                if show_plot:
                    plt.show()
