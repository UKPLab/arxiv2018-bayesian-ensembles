'''
Created on Nov 1, 2016

@author: Melvin Laux
'''

import baselines as bsl
import ConfigParser
import os, sys
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt


class Experiment(object):
    '''
    classdocs
    '''
    
    generator = None
    
    config_file = None
    
    param_values = None
    param_idx = None
    param_fixed = None
    
    param_names = ['acc_bias','miss_bias','short_bias']
    
    method = None
    
    num_runs = None
    
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
        param_values = np.array(eval(parameters['values'].split('#')[0].strip()))
        param_fixed = eval(parameters['fixed'].split('#')[0].strip())
        
        self.param_values = np.ones((len(param_values), 3))
        
        self.method = parameters['method'].split('#')[0].strip()
        
        idx = 0
        
        while idx < 3:
            if idx == self.param_idx:
                self.param_values[:,idx] = param_values
            else:
                self.param_values[:,idx] *= param_fixed.pop(0)
                
            idx += 1
        
        self.num_runs = int(parameters['num_runs'].split('#')[0].strip())
        
        # set up output
        parameters = dict(parser.items('output'))
        self.output_dir = parameters['output_dir']
        self.save_results = eval(parameters['save_results'].split('#')[0].strip())
        self.save_plots = eval(parameters['save_plots'].split('#')[0].strip())
        self.show_plots = eval(parameters['show_plots'].split('#')[0].strip())
        
        
    def single_run(self, param_values):
        
        scores = np.zeros((param_values.shape[0], 4))
       
        for i in xrange(param_values.shape[0]):
            params = param_values[i,:]
            self.generator.init_crowd_model(params[0], params[1], params[2])
            ground_truth, annotations, _ = self.generator.generate_dataset(crowd_size=3)
            
            agg = None
            
            if self.method == 'majority':
                mv = bsl.majority_voting.MajorityVoting(ground_truth, annotations, 3)
                agg,_ = mv.vote()
            elif self.method == 'clustering':
                cl = bsl.clustering.Clustering(ground_truth, annotations)
                agg = cl.run_kde()
            else:
                sys.exit('unknown method: ' + self.method)
            
            scores[i,1],scores[i,2],scores[i,3],_ = skm.precision_recall_fscore_support(ground_truth[:,1], agg, average='macro')
            scores[i,0] = skm.accuracy_score(ground_truth[:,1], agg)
        
        return scores
    
    
    def run(self, param_values, num_runs, show_plot=False, save_plot=False, save_results=False, output_dir='output/'):
        
        results = np.zeros((param_values.shape[0], 4, num_runs))
        print 'Running experiment...'
        for i in xrange(num_runs):      
            results[:,:,i] = self.single_run(param_values) 
            
        if save_results:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            print 'Saving results...'
            np.savetxt(output_dir + 'results.csv', np.mean(results,2))

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
            
        plt.errorbar(self.param_values[:, self.param_idx], np.mean(results[:,0,:], 1), yerr=np.var(results[:,0,:], 1), label='accuracy')
        plt.errorbar(self.param_values[:, self.param_idx], np.mean(results[:,1,:], 1), yerr=np.var(results[:,1,:], 1), label='precision')
        plt.errorbar(self.param_values[:, self.param_idx], np.mean(results[:,2,:], 1), yerr=np.var(results[:,2,:], 1), label='recall')
        plt.errorbar(self.param_values[:, self.param_idx], np.mean(results[:,3,:], 1), yerr=np.var(results[:,3,:], 1), label='f1-score')
        
        plt.legend(loc=0)
        
        plt.title('parameter influence')
        plt.ylabel('score')
        plt.xlabel(self.param_names[self.param_idx])
        
        if save_plot:
            print 'Saving plot...'
            plt.savefig(self.output_dir + 'plot.png') 
        
        if show_plot: 
            plt.show()
        
        
        