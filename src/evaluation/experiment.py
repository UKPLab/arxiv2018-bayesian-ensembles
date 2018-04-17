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
from baselines.hmm import HMM_crowd
from baselines.util import crowd_data, crowdlab, instance

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
    SCORE_NAMES = ['accuracy', 'precision', 'recall', 'f1-score', 'auc-score', 'cross-entropy-error', 'count error', 
                   'number of invalid labels', 'mean length error']
    
    generate_data= False
    
    methods = None
        
    bac_alpha0 = None
    bac_nu0 = None
    
    exclusions = {}

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
        
    
    def run_methods(self, annotations, ground_truth, doc_start, param_idx, anno_path, return_model=False):
        
        scores = np.zeros((len(self.SCORE_NAMES), len(self.methods)))
        predictions = -np.ones((annotations.shape[0], len(self.methods)))
        probabilities = -np.ones((annotations.shape[0], self.num_classes, len(self.methods)))
        
        model = None # pass out to other functions if required
        
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
                
                #devnull = open(os.devnull, 'w')
                subprocess.call(['java', '-jar', './MACE/MACE.jar', '--distribution', '--prefix', './output/data/mace', anno_path])#, stdout = devnull, stderr = devnull)
                
                result = np.genfromtxt('./output/data/mace.prediction')
                    
                agg = result[:, 0]
                    
                probs = np.zeros((ground_truth.shape[0], self.num_classes))
                for i in xrange(result.shape[0]):
                    for j in xrange(0, self.num_classes * 2, 2):
                        probs[i, int(result[i, j])] = result[i, j + 1]  
                
            if self.methods[method_idx] == 'ibcc':
                
                alpha0 = np.ones((self.num_classes, self.num_classes, 10))
                alpha0[:, :, 5] = 2.0
                alpha0 += np.eye(self.num_classes)[:, :, None]
                nu0 = np.ones(self.num_classes, dtype=float)
                
                ibc = ibcc.IBCC(nclasses=self.num_classes, nscores=self.num_classes, nu0=nu0, alpha0=alpha0)
                probs = ibc.combine_classifications(annotations, table_format=True) # posterior class probabilities
                agg = probs.argmax(axis=1) # aggregated class labels
                
            if self.methods[method_idx] == 'bac':
                L = self.num_classes
                if L == 7:
                    inside_labels = [0, 3, 5]
                else:
                    inside_labels = [0]
                    
                alg = bac.BAC(L=L, K=annotations.shape[1], inside_labels=inside_labels, alpha0=self.bac_alpha0, 
                              nu0=self.bac_nu0, exclusions=self.exclusions, before_doc_idx=-1)
                alg.verbose = True
                probs, agg = alg.run(annotations, doc_start)
                model = alg
                
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
                hc = HMM_crowd(self.num_classes, 1, data, None, None, vb=[0.1, 0.1], ne=1)
                hc.init(init_type = 'dw', wm_rep='cv2', dw_em=5, wm_smooth=0.1)
                hc.em(20)
                hc.mls()
                agg = np.array(hc.res).flatten() 
                agg = np.concatenate(hc.res)[:,None]
                probs = []
                for sentence_post_arr in hc.sen_posterior:
                    for tok_post_arr in sentence_post_arr:
                        probs.append(tok_post_arr)
                probs = np.array(probs)
            
            scores[:,method_idx][:,None] = self.calculate_scores(agg, ground_truth.flatten(), probs, doc_start)
            predictions[:, method_idx] = agg.flatten()
            probabilities[:,:,method_idx] = probs
            
            print '...done'
            
        if return_model:
            return scores, predictions, probabilities, model
        else:
            return scores, predictions, probabilities
    
    
    def calculate_scores(self, agg, gt, probs, doc_start):
        
        result = -np.ones((9,1))        
        result[7] = metrics.num_invalid_labels(agg, doc_start)        
        
        agg = agg[gt!=-1]
        probs = probs[gt!=-1]
        doc_start = doc_start[gt!=-1]
        gt = gt[gt!=-1]
                        
        if self.postprocess:
            agg = data_utils.postprocess(agg, doc_start)
        
        result[0] = skm.accuracy_score(gt, agg)
        
        print 'Plotting confusion matrix for errors: '
        
        nclasses = probs.shape[1]
        conf = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                conf[i, j] = np.sum((gt==i).flatten() & (agg==j).flatten())
                
            #print 'Acc for class %i: %f' % (i, skm.accuracy_score(gt==i, agg==i))
        print conf
        
        result[1] = skm.precision_score(gt, agg, average='macro')
        result[2] = skm.recall_score(gt, agg, average='macro')
        result[3] = skm.f1_score(gt, agg, average='macro')
        
        auc_score = 0
        for i in xrange(probs.shape[1]):
            auc_i = skm.roc_auc_score(gt==i, probs[:, i]) 
            #print 'AUC for class %i: %f' % (i, auc_i)
            auc_score += auc_i * np.sum(gt==i)
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
                doc_start, gt, annos = self.generator.read_data_file(data_path + 'full_data.csv')                
                # run methods
                results[param_idx,:,:,run_idx], preds, probabilities = self.run_methods(annos, gt, doc_start, param_idx, data_path + 'annotations.csv')
                # save predictions
                np.savetxt(data_path + 'predictions.csv', preds)
                # save probabilities
                probabilities.dump(data_path + 'probabilites')
                
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
        
        if not glob.glob(os.path.join(self.output_dir, 'data/*')):
            print 'No data found at specified path! Exiting!'
            return
        else:
            return self.run_exp()
        
        
    def make_plot(self, x_vals, y_vals, x_ticks_labels, ylabel):
        
        styles = ['-', '--', '-.', ':']
        markers = ['o', 'v', 's', 'p', '*']
        
        for j in xrange(len(self.methods)):
            plt.plot(x_vals, np.mean(y_vals[:, j, :], 1), label=self.methods[j], ls=styles[j%4], marker=markers[j%5])
                  
        plt.legend(loc='best')
        
        plt.title('parameter influence')
        plt.ylabel(ylabel)
        plt.xlabel(self.PARAM_NAMES[self.param_idx])
        plt.xticks(x_vals, x_ticks_labels)
        plt.grid(True)
        plt.grid(True, which='Minor', axis='y')
        
        if np.min(y_vals) < 0:
            plt.ylim([np.min(y_vals), np.max([1, np.max(y_vals)])])
        else:
            plt.ylim([0, np.max([1, np.max(y_vals)])])
    
    
    def plot_results(self, results, show_plot=False, save_plot=False, output_dir='/output/', score_names=None):
        
        if score_names is None: 
            score_names = self.SCORE_NAMES
        
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
            
        for i in xrange(len(score_names)):  
            self.make_plot(x_vals, results[:,i,:,:], x_ticks_labels, score_names[i])
        
            if save_plot:
                print 'Saving plot...'
                plt.savefig(output_dir + 'plot_' + score_names[i] + '.png') 
                plt.clf()
        
            if show_plot:
                plt.show()
                
            if i == 5:
                self.make_plot(x_vals, results[:,i,:,:], x_ticks_labels, score_names[i])
                plt.ylim([0,1])
                
                if save_plot:
                    print 'Saving plot...'
                    plt.savefig(output_dir + 'plot_' + score_names[i] + '_zoomed' + '.png') 
                    plt.clf()
        
                if show_plot:
                    plt.show()