import numpy as np
scores = np.array([0,1,2,3,4])
nScores = len(scores)
nClasses = 5
inputFile = "./data/crowdlabels_table5.csv"
tableFormat = True
outputFile = "./data/test.out"
confMatFile = "./data/test_ibcc.mat"
goldFile = "./data/gold5.csv"
nu0 = np.array([50,50,50,50,50])
alpha0 = np.array([[2, 1, 1, 1, 1], [1, 2, 1, 1, 1], [1, 1, 2, 1, 1], [1, 1, 1, 2, 1], [1, 1, 1, 1, 2]])
uselowerbound = True
