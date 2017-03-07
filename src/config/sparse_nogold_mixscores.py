import numpy as np
scores = np.array([10,3])
nScores = len(scores)
nClasses = 2
inputFile = "./data/crowdlabels_sparse_scoremix.csv"
outputFile = "./data/test.out"
confMatFile = "./data/test_ibcc.mat"
nu0 = np.array([50,50])
alpha0 = np.array([[2, 1], [1, 2]])
