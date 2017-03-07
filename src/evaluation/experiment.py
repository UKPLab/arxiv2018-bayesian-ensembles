'''
Created on Nov 1, 2016

@author: Melvin Laux
'''

from src.baselines import ibcc, clustering, majority_voting
import ConfigParser
import os, subprocess
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
        
        self.method = eval(parameters['method'].split('#')[0].strip())
        
        print self.method
        
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
        
        scores = np.zeros((param_values.shape[0], 4, len(self.method)))
       
        for i in xrange(param_values.shape[0]):
            params = param_values[i,:]
            self.generator.init_crowd_model(params[0], params[1], params[2])
            ground_truth, annotations, _ = self.generator.generate_dataset(crowd_size=10, save_to_file=True)
            
            agg = None
            
            idx = 0
            
            if 'majority' in self.method:
                mv = majority_voting.MajorityVoting(ground_truth, annotations, 3)
                agg,_ = mv.vote()
                
                scores[i,1,idx],scores[i,2,idx],scores[i,3,idx],_ = skm.precision_recall_fscore_support(ground_truth[:,1], agg, average='macro')
                scores[i,0,idx] = skm.accuracy_score(ground_truth[:,1], agg)
                idx += 1
            
            if 'clustering' in self.method:
                cl = clustering.Clustering(ground_truth, annotations)
                agg = cl.run()
            
                scores[i,1,idx],scores[i,2,idx],scores[i,3,idx],_ = skm.precision_recall_fscore_support(ground_truth[:,1], agg, average='macro')
                scores[i,0,idx] = skm.accuracy_score(ground_truth[:,1], agg)
                idx += 1
                
            if 'mace' in self.method:
                subprocess.call(['java', '-jar', 'MACE/MACE.jar', 'output/data/annotations.csv'])
                
                agg = np.genfromtxt('prediction', delimiter=',')
            
                scores[i,1,idx],scores[i,2,idx],scores[i,3,idx],_ = skm.precision_recall_fscore_support(ground_truth[:,1], agg, average='macro')
                scores[i,0,idx] = skm.accuracy_score(ground_truth[:,1], agg)
                idx += 1
                
            if 'ibcc' in self.method:
                
                alpha0 = np.ones((3,3,10))
                alpha0[:, :, 5] = 2.0
                alpha0[np.arange(3),np.arange(3),:] += 1.0
                #alpha0[np.arange(3),np.arange(3),-1] += 20

                
                nu0 = np.array([1,1,1], dtype=float)
                
                ibc = ibcc.IBCC(nclasses=3,nscores=3,nu0=nu0, alpha0=alpha0)
                agg = ibc.combine_classifications(annotations,table_format=True).argmax(axis=1)
                
                scores[i,1,idx],scores[i,2,idx],scores[i,3,idx],_ = skm.precision_recall_fscore_support(ground_truth[:,1], agg, average='macro')
                scores[i,0,idx] = skm.accuracy_score(ground_truth[:,1], agg)
                idx += 1
                
        
        return scores
    
    
    def run(self, param_values, num_runs, show_plot=False, save_plot=False, save_results=False, output_dir='output/'):
        
        results = np.zeros((param_values.shape[0], 4, len(self.method), num_runs))
        print 'Running experiment...'
        for i in xrange(num_runs):      
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
            
        score_names = ['accuracy', 'precision','recall','f1-score']
            
        for i in xrange(4):    
            for j in xrange(len(self.method)):
                plt.errorbar(self.param_values[:, self.param_idx], np.mean(results[:,i,:,j], 1), yerr=np.var(results[:,i,:,j], 1), label=self.method[j])
            
            plt.legend(loc=0)
        
            plt.title('parameter influence')
            plt.ylabel(score_names[i])
            plt.xlabel(self.param_names[self.param_idx])
        
            if save_plot:
                print 'Saving plot...'
                plt.savefig(self.output_dir + 'plot_' + score_names[i] + '.png') 
        
            if show_plot: 
                plt.show()
        
        
        