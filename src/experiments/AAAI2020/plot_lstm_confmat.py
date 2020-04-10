import numpy as np
import os
import glob

L = 3
IOB_idx = [0, 1, 2, 0, 2, 0, 2, 0, 2]
ntypes = [4, 1, 4]

# inputs: list of filenames containing saved LSTM confusion matrices from BSC-Seq
# outputs: csv file containing per-class accuracies obtained from the tables as an unweighted average over previous labels/current labels

files = glob.glob(os.path.join("data/LSTM_worker_models/", "*.npy"))
print(files)

accs = []

for f in files:
    cm = np.load(f)
    acc_by_class = np.zeros(3)

    for truelabel in range(L):
        for prev in range(L):
            acc_by_class[IOB_idx[truelabel]] += cm[truelabel, truelabel, prev] / float(np.sum(cm[truelabel, :, prev]))

    acc_by_class /= L * np.array(ntypes).astype(float)

    accs.append(acc_by_class)

np.savetxt('./LSTM_accuracy_summary_PICO_task2.csv', np.array(accs), fmt='%.3f')
