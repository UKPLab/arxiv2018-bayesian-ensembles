'''
Created on 29 Sep 2014

@author: edwin
'''

import pickle, logging
import numpy as np
from scipy.sparse import coo_matrix

class DataHandler(object):
    '''
    classdocs
    '''

    #default values ####################
    scores = np.array([3, 4])
    K = 0
    N = 0
    nclasses = 2
    nu0 = np.array([50.0, 50.0])
    alpha0 = np.array([[2, 1], [1, 2]])  

    ####################################
    
    uselowerbound = False

    crowdlabels = None
    table_format = False
    
    targetidxmap = None
    targetidxs = None
    max_targetid = 0
    trainids = None

    goldlabels = None
    goldsubtypes = None
    
    output_file = None
    confmat_file = None
    input_file = None
    gold_file = None
    hyperparam_file = None

    def __init__(self):
        '''
        Constructor
        '''
        
    def create_target_idx_map(self):
        self.max_targetid = np.max(self.targetidxs) # largest original ID value
        blanks = np.zeros(len(self.targetidxs)) # only need 1D so set all to zero
        idxList = range(len(self.targetidxs)) # new local idxs
        tIdxMap = coo_matrix(( idxList, (self.targetidxs,blanks)), shape=(self.max_targetid+1,1) )       
        self.N = len(self.targetidxs)
        self.targetidxmap = tIdxMap.tocsr() # maps Original IDs to new local idxs
        
    def loadCrowdLabels(self, scores):   
        '''
        Loads labels from crowd in sparse list format, i.e. 3 columns, classifier ID,
        object ID, score.
        '''
        pyFileExists = False
        try:
            with open(self.input_file+'.dat','r') as inFile:
                crowdLabels, self.targetidxs, K = pickle.load(inFile)
                pyFileExists = True
        except Exception:
            logging.info('Will try to load a CSV file...')
        
            crowdLabels = np.genfromtxt(self.input_file, delimiter=',', \
                                    skip_header=1,usecols=[0,1,2])
    
            self.targetidxs, crowdLabels[:,1] = np.unique(crowdLabels[:,1],return_inverse=True)
            kIdxs, crowdLabels[:,0] = np.unique(crowdLabels[:,0],return_inverse=True)
            K = len(kIdxs)
            
        unmappedScores = np.round(crowdLabels[:,2])
        for i,s in enumerate(scores):
            print np.sum(unmappedScores==s)
            crowdLabels[(unmappedScores==s),2] = i
        
        self.create_target_idx_map()
        self.crowdlabels = crowdLabels
        self.K = K
        print crowdLabels.shape
        if not pyFileExists:
            try:
                with open(self.input_file+'.dat', 'wb') as outFile:
                    pickle.dump((crowdLabels,self.targetidxs,K), outFile)
            except Exception:
                logging.error('Could not save the input data as a Python object file.')
        
    def loadCrowdTable(self, scores):
        '''
        Loads crowd labels in a table format
        '''
        unmappedScores = np.round(np.genfromtxt(self.input_file, delimiter=','))
        self.K = unmappedScores.shape[1]
        self.targetidxs = np.arange(unmappedScores.shape[0])
        self.create_target_idx_map()
                   
        self.crowdlabels = np.empty((self.N,self.K))
        self.crowdlabels[:,:] = np.nan
        for i,s in enumerate(scores):
            self.crowdlabels[unmappedScores==s] = i
        
    def loadGold(self, classLabels=None, secondaryTypeCol=-1):   
        
        import os.path
        if not os.path.isfile(self.gold_file):
            logging.info('No gold labels found -- running in unsupervised mode.')
            self.goldlabels = np.zeros(self.N) -1
            return
        
        if secondaryTypeCol>-1:
            useCols=[0,1,secondaryTypeCol]
        else:
            useCols=[0,1]
            
        try:
            gold = np.genfromtxt(self.gold_file, delimiter=',', skip_header=0,usecols=useCols,invalid_raise=True)
        except Exception:
            gold = np.genfromtxt(self.gold_file, delimiter=',', skip_header=0)
            
        if np.any(np.isnan(gold[0])): #skip header if necessary
            gold = gold[1:,:]
        logging.debug("gold shape: " + str(gold.shape))
        
        if len(gold.shape)==1 or gold.shape[1]==1: #position in this list --> id of data point
            goldLabels = gold
            goldIdxs = np.arange(len(goldLabels))
            missing_idxs = [i for i in goldIdxs if not i in self.targetidxs]
            if len(missing_idxs):
                #There are more gold labels than data points with crowd labels.
                self.targetidxs = np.concatenate((self.targetidxs,missing_idxs))
                self.create_target_idx_map()
        else: # sparse format: first column is id of data point, second column is gold label value
            
            #map the original idxs to local idxs
            # -- Commented out because we shouldn't remove data points that have no crowd labels
            #valid_gold_idxs = np.argwhere(gold[:,0]<=self.max_targetid)
            #gold = gold[valid_gold_idxs.reshape(-1),:]
            goldIdxs = gold[:,0]
            # -- Instead, we must append the missing indexes to the list of targets
            missing_idxs = [i for i in goldIdxs if not i in self.targetidxs]
            if len(missing_idxs):
                self.targetidxs = np.concatenate((self.targetidxs,missing_idxs))
                self.create_target_idx_map()
            
            #map the IDs to their local index values
            goldIdxs = self.targetidxmap[goldIdxs,0].todense()
            
            #create an array for gold for all the objects/data points in this test set
            goldLabels = np.zeros(self.N) -1
            goldLabels[goldIdxs] = gold[:,1]
            
            #if there is secondary type info, create a similar array for this
            if secondaryTypeCol>-1:
                goldTypes = np.zeros(self.N)
                goldTypes[goldIdxs] = gold[:,2]
                goldTypes[np.isnan(goldTypes)] = 0 #some examples may have no type info
                goldTypes[goldLabels==-1] = -1 #negative examples have type -1
              
        if classLabels:
            #convert text to class IDs
            for i in range(gold.shape[0]):
                classIdx = np.where(classLabels==goldLabels[i])
                if classIdx:
                    goldLabels[i] = classIdx
                else:
                    goldLabels[i] = -1

        self.goldlabels = goldLabels        
        if secondaryTypeCol>-1:
            self.goldsubtypes = goldTypes
        
    def map_predictions_to_original_IDs(self, predictions, return_array=True):
        rows = np.tile(self.targetidxs.reshape((self.N,1)), (1,self.nclasses)).flatten()
        cols = np.tile(np.arange(self.nclasses), (self.N,1)).flatten()
        data = predictions.flatten()
        mapped_predictions = coo_matrix((data, (rows,cols)), shape=(self.max_targetid+1, self.nclasses))
        if return_array:
            mapped_predictions = mapped_predictions.toarray()
        return mapped_predictions
                
    def loadData(self, configFile):
        
        testid="unknowntest"
        
        #Defaults that will usually be overwritten by project config
        tableFormat = False
        #columns in input file:
        # 0 = agent/worker/volunteer ID
        # 1 = object ID
        # 2 = scores given to the object
        
        #columns in gold file
        # 0 = object ID
        # 1 = class label
        scores = self.scores
        
        classLabels = None
      
        trainIds = None #IDs of targets that should be used as training data. Optional 
        
        #column index of secondary type information about the data points stored in the gold file. 
        #-1 means no such info
        goldTypeCol = -1
        
        def translate_gold(gold):
            return gold
        
        outputFile = './output/output_%s.csv'
        confMatFile = ''#'./output/confMat.csv'
        hyperparam_file = ''
        inputFile = './data/input.csv'
        goldFile = ''#./data/gold.csv'
        
        nClasses = 2
        nu0 = self.nu0
        alpha0 = self.alpha0 
        uselowerbound = self.uselowerbound
        
        #read configuration
        with open(configFile, 'r') as conf:
            configuration = conf.read()
            exec(configuration)

        print hyperparam_file
        print testid
        try:
            self.output_file = outputFile % testid
            self.confmat_file = confMatFile % testid
            self.hyperparam_file = hyperparam_file  % testid
        except TypeError:
            self.output_file = outputFile
            self.confmat_file = confMatFile
            self.hyperparam_file = hyperparam_file
        self.input_file = inputFile
        self.gold_file = goldFile
        self.scores = scores
        self.nclasses = nClasses
        self.nu0 = nu0
        self.alpha0 = alpha0
        self.uselowerbound = uselowerbound
    
        #load labels from crowd
        if tableFormat:
            self.loadCrowdTable(scores)
        else:
            self.loadCrowdLabels(scores)
        
        #load gold labels if present
        self.loadGold(classLabels, goldTypeCol)
            
        self.goldlabels = translate_gold(self.goldlabels)
        
        #map the training IDs to our local indexes
        if trainIds != None:
            self.trainids = self.targetidxmap[trainIds,0].todense()
        
        self.table_format = tableFormat
            
    def save_targets(self, pT):
        #write predicted class labels to file
        logging.info('writing results to file')
        logging.debug('Posterior matrix: ' + str(pT.shape))
        tIdxs = np.reshape(self.targetidxs, (len(self.targetidxs),1))
        logging.debug('Target indexes: ' + str(tIdxs.shape))    
        np.savetxt(self.output_file, np.concatenate([tIdxs, pT], 1))
    
    def save_pi(self, alpha, nclasses, nscores):
        #write confusion matrices to file if required
        if self.confmat_file is None or self.confmat_file=='':
            return

        # the defaults which existed before they were read in as param
        # nscores = self.scores.size
        # nclasses = self.nclasses
    
        logging.info('writing confusion matrices to file')
        pi = np.zeros(alpha.shape)
        for l in range(nscores):
            pi[:,l,:] = alpha[:,l,:]/np.sum(alpha,1)
        
        flatPi = pi.reshape(1, nclasses*nscores, alpha.shape[2])
        flatPi = np.swapaxes(flatPi, 0, 2)
        flatPi = flatPi.reshape(alpha.shape[2], nclasses*nscores)
        np.savetxt(self.confmat_file, flatPi, fmt='%1.3f')
        
    def save_hyperparams(self, alpha, nu):    
        if self.hyperparam_file is None or self.hyperparam_file=='':
            return
        
        nscores = self.scores.size
        logging.info('writing hyperparameters to file')
        flatalpha = np.swapaxes(alpha, 0, 2)
        flatalpha = flatalpha.flatten()
        flatalpha = flatalpha.reshape(alpha.shape[2], self.nclasses*nscores)
        
        np.savetxt(self.hyperparam_file, flatalpha, fmt='%1.3f')
        
        nu = nu.flatten()
        others = []
        for nuj in nu:
            others.append(nuj)
        np.savetxt(self.hyperparam_file+"_others.csv", others, fmt='%1.3f')
