from bayesian_combination.bsc import BSC
import numpy as np
import pandas as pd

path = "../../data/eyetracking/"

# TODO: how to deal with NA values? (There are none in part 1, but in part 3)
C = np.loadtxt(open(path + "part1_annos.csv", "r"), delimiter=",", skiprows=1)

doc_start = np.loadtxt(open(path + "part1_doc_start.csv", "r"), delimiter=",")

with open(path + "part1_text.csv", "r") as f:
    lines = f.read().splitlines()
features = lines

print(C.shape)
print(doc_start.shape)
print(len(features))

# TODO: how to transform the data into bins? Placeholder solution creating 5 equally sized bins
categorical_values, bins = pd.qcut(C.flatten(), 5, labels=False, retbins=True)
categorical_matrix = categorical_values.reshape(C.shape)
print(bins)

# Transform continuous data into categorical values based on bins
# categorical_matrix = np.digitize(C, bins, right=True)  # this doesn't work because it treats the left-hand bin edge
# as exclusive, but there is one value equal to the bin edge... so digitize ends up creating 6 categories!
print(categorical_matrix[0:15])
print(features[0:15])

model = BSC(L=5, K=14, tagging_scheme="none")
model.fit_predict(categorical_matrix, doc_start=doc_start, features=np.asarray(features), dev_sentences=[], gold_labels=None)