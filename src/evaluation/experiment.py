'''
Created on Nov 1, 2016

@author: Melvin Laux
'''

from src.baselines import ibcc, clustering, majority_voting
from src.algorithm import bac
import ConfigParser
import os, subprocess
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt
from numpy.distutils.system_info import agg2_info


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
    
    param_names = ['acc_bias','miss_bias','short_bias','num_docs', 'doc_length','crowd_size']
    score_names = ['accuracy', 'precision','recall','f1-score','auc-score', 'cross-entropy-error','count error', 'number of invalid labels', 'mean length error']
    
    method = None
    
    num_runs = None
    doc_length = None
    crowd_size = None
    num_docs = None
    
    num_classes = None
    
    output_dir = None
    
    save_results = False
    save_plots = False
    show_plots = False
    

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
        
        self.acc_bias = int(parameters['acc_bias'].split('#')[0].strip())
        self.miss_bias = int(parameters['miss_bias'].split('#')[0].strip())
        self.short_bias = int(parameters['short_bias'].split('#')[0].strip())
        
        
        self.method = eval(parameters['method'].split('#')[0].strip())
        
        print self.method
                
        
        self.num_runs = int(parameters['num_runs'].split('#')[0].strip())
        self.doc_length = int(parameters['doc_length'].split('#')[0].strip())
        self.crowd_size = int(parameters['crowd_size'].split('#')[0].strip())
        self.num_classes = int(parameters['num_classes'].split('#')[0].strip())
        self.num_docs = int(parameters['num_docs'].split('#')[0].strip())
        
        # set up output
        parameters = dict(parser.items('output'))
        self.output_dir = parameters['output_dir']
        self.save_results = eval(parameters['save_results'].split('#')[0].strip())
        self.save_plots = eval(parameters['save_plots'].split('#')[0].strip())
        self.show_plots = eval(parameters['show_plots'].split('#')[0].strip())
        
        
    def single_run(self, param_values):
        
        scores = np.zeros((self.param_values.shape[0], len(self.score_names), len(self.method)))
       
        for i in xrange(param_values.shape[0]):
            # update tested parameter
            if self.param_idx == 0:
                self.acc_bias = self.param_values[i]
            elif self.param_idx == 1:
                self.miss_bias = self.param_values[i]
            elif self.param_idx == 2:
                self.short_bias = self.param_values[i]
            elif self.param_idx == 3:
                self.num_docs = self.param_values[i]
            elif self.param_idx == 4:
                self.doc_length = self.param_values[i]
            elif self.param_idx == 5:
                self.crowd_size = self.param_values[i]
            else:
                print 'Encountered invalid test index!'
                
            self.generator.init_crowd_model(self.acc_bias, self.miss_bias, self.short_bias)
            ground_truth, annotations, _ = self.generator.generate_dataset(num_docs=self.num_docs, doc_length=self.doc_length, crowd_size=self.crowd_size, save_to_file=True)
            
            agg = None
            
            for method_idx in xrange(len(self.method)):
            
                if self.method[method_idx] == 'majority':
                    mv = majority_voting.MajorityVoting(annotations, self.num_classes)
                
                    agg, probs = mv.vote()
            
                if self.method[method_idx] == 'clustering':
                    cl = clustering.Clustering(ground_truth, annotations)
                    agg = cl.run()
                
                    probs = np.zeros((ground_truth.shape[0], self.num_classes))
                
                    for k in xrange(ground_truth.shape[0]):
                        probs[k,int(agg[k])] = 1      
                
                if self.method[method_idx] == 'mace':
                    subprocess.call(['java', '-jar', 'MACE/MACE.jar', '--prefix', 'output/data/mace', 'output/data/annotations.csv'])
                
                    agg = np.genfromtxt('output/data/mace.prediction', delimiter=',')
                
                    probs = np.zeros((ground_truth.shape[0], self.num_classes))
                
                if self.method[method_idx] == 'ibcc':
                
                    alpha0 = np.ones((self.num_classes,self.num_classes,10))
                    alpha0[:, :, 5] = 2.0
                    alpha0[np.arange(3),np.arange(3),:] += 1.0
                
                    nu0 = np.array([1,1,1], dtype=float)
                
                    ibc = ibcc.IBCC(nclasses=3, nscores=3, nu0=nu0, alpha0=alpha0)
                    agg = ibc.combine_classifications(annotations, table_format=True).argmax(axis=1)
                
                    probs = np.zeros((ground_truth.shape[0], self.num_classes))
                
                if self.method[method_idx] == 'bac':
                
                    alg = bac.BAC(L=3, K=annotations.shape[1])
                
                    doc_start = np.zeros((annotations.shape[0],1))
                    doc_start[0] = 1
                
                    E_t, most_prob = alg.run(annotations+1, doc_start)
                
                    agg = E_t.argmax(axis=1)
                    
                self.update_scores(scores, i, method_idx, agg, ground_truth, E_t)
        
        return scores
    
    
    def update_scores(self, scores, param_idx, method_idx, agg, ground_truth, probs):
        
        scores[param_idx, 0, method_idx] = skm.accuracy_score(ground_truth[:,1], agg)
        scores[param_idx, 1, method_idx],scores[param_idx, 2, method_idx],scores[param_idx, 3, method_idx],_ = skm.precision_recall_fscore_support(ground_truth[:,1], agg, pos_label=0, average='macro')
        auc_score = skm.roc_auc_score(ground_truth[:,1]==0, agg==0)*np.sum(ground_truth[:,1]==0)
        auc_score += skm.roc_auc_score(ground_truth[:,1]==1, agg==1)*np.sum(ground_truth[:,1]==1)
        auc_score += skm.roc_auc_score(ground_truth[:,1]==2, agg==2)*np.sum(ground_truth[:,1]==2)
        scores[param_idx, 4, method_idx] = auc_score/float(ground_truth.shape[0])
        scores[param_idx, 5, method_idx] = skm.log_loss(ground_truth[:,1], probs, eps=1e-100)
        scores[param_idx, 6, method_idx] = abs_count_error(agg, ground_truth[:,1])
        scores[param_idx, 7, method_idx] = num_invalid_labels(agg)
        scores[param_idx, 8, method_idx] = mean_length_error(agg, ground_truth[:,1])
    
    
    def run(self, param_values, num_runs, show_plot=False, save_plot=False, save_results=False, output_dir='output/'):
        
        results = np.zeros((param_values.shape[0], len(self.score_names), len(self.method), num_runs))
        print 'Running experiment...'
        for i in xrange(num_runs):   
            print 'Run number:', i   
            results[:,:,:,i] = self.single_run(param_values) 
            
        if save_results:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            print 'Saving results...'
            np.savetxt(output_dir + 'results.csv', np.mean(results,3)[:,:,0])

        if show_plot or save_plot:
            self.plot_results(results,show_plot,save_plot,output_dir)
        
        return results
    
    
    def run_config(self):
        if self.config_file == None:
            raise RuntimeError('No config file was specified.')
        
        return self.run(self.param_values, self.num_runs, self.show_plots, self.save_plots, self.save_results, self.output_dir)
    
    
    def plot_results(self, results, show_plot=False, save_plot=False, output_dir='/output/'):
        
        if save_plot and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for i in xrange(len(self.score_names)):    
            for j in xrange(len(self.method)):
                plt.errorbar(self.param_values, np.mean(results[:,i,j,:], 1), yerr=np.std(results[:,i,j,:], 1), label=self.method[j])
                
            
            plt.legend(loc=0)
        
            plt.title('parameter influence')
            plt.ylabel(self.score_names[i])
            plt.xlabel(self.param_names[self.param_idx])
            
            plt.ylim([0,np.max([1,np.max(results[:,i,:,:])])])
        
            if save_plot:
                print 'Saving plot...'
                plt.savefig(self.output_dir + 'plot_' + self.score_names[i] + '.png') 
        
            if show_plot:
                plt.show()

'''

'''        
def abs_count_error(pred, target):
    return np.abs(len(pred==2) - len(target==2))

'''

'''        
def num_invalid_labels(pred):    
    return len(np.where(pred[np.r_[0,np.where(pred[:-1]==1)[0] + 1]]==0)[0])
     

'''

'''        
def mean_length_error(pred, target):
    # delete 'O'-tokens
    pred = np.delete(pred, np.where(pred==1)[0])
    target = np.delete(target, np.where(target==1)[0])

    return np.abs(np.mean(np.array([len(x) for x in np.split(target, np.where(target==2)[0])[1:]])) - np.mean(np.array([len(x) for x in np.split(pred, np.where(pred==2)[0])[1:]])))


            