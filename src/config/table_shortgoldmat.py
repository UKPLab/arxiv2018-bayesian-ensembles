import numpy as np
scores = np.array([0,1])
nScores = len(scores)
nClasses = 2
inputFile = "./data/crowdlabels_table_mixed.csv"
tableFormat = True
outputFile = "./data/test.out"
confMatFile = "./data/test_ibcc.mat"
goldFile = "./data/goldmat_mixed.csv"
nu0 = np.array([50,50])
alpha0 = np.array([[2, 1], [1, 2]])
