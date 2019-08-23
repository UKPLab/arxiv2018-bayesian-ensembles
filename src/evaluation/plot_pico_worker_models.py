'''

Train each variant of BSC on all gold labels.
Extract the worker models from each variant.
Cluster into ~5 groups.
Plot the centroid or mean of the 5 groups.

Created on April 27, 2018

@author: Edwin Simpson
'''
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from bsc.bsc import BSC
import data.load_data as load_data
import numpy as np
from sklearn.cluster import MiniBatchKMeans

output_dir = '../../data/bayesian_sequence_combination/output/bio_workers/'

# debug with subset -------
s = None #1000
gt, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev = load_data.load_biomedical_data(False, s)

# # get all the gold-labelled data together
goldidxs_dev = gt_task1_dev != -1
gt[goldidxs_dev] = gt_task1_dev[goldidxs_dev]

nu0_factor = 0.1
alpha0_diags = 0.1
alpha0_factor = 0.1

plot_integrated=False

if plot_integrated:
    worker_models = [
        'seq'
    ]

    # corresponding list of data models
    data_models = [
        ['IF',  'LSTM']
    ]
else:
    worker_models = [
                    # 'acc',
                    # 'vec',
                    # 'ibcc',
                    # 'mace',
                    'seq',
    ]

    data_models = [
        # ['IF'],
        # ['IF'],
        # ['IF'],
        # ['IF'],
        ['IF'],
    ]

inside_labels = [0]
outside_labels = [1]
begin_labels = [2]

L = 3

confmats = {}

for w, worker_model in enumerate(worker_models):
# matrices are repeated for the different annotators/previous label conditions inside the BAC code itself.
    if worker_model == 'seq' or worker_model == 'ibcc' or worker_model == 'vec':

        f = alpha0_factor / ((L - 1) / 2)
        d = alpha0_diags + alpha0_factor - alpha0_factor

        alpha0 = f * np.ones((L, L)) + d * np.eye(L)

    elif worker_model == 'mace':
        f = alpha0_factor  # / ((L-1)/2)
        alpha0 = alpha0_factor * np.ones((2 + L))
        alpha0[1] += alpha0_diags  # diags are bias toward correct answer

    elif worker_model == 'acc':
        alpha0 = alpha0_factor * np.ones((2))
        alpha0[1] += alpha0_diags  # diags are bias toward correct answer

    data_model = data_models[w]

    model = BSC(L=L, K=annos.shape[1], inside_labels=inside_labels, outside_labels=outside_labels,
                beginning_labels=begin_labels, alpha0_diags=alpha0_diags, alpha0_factor=alpha0_factor,
                beta0_factor=nu0_factor, before_doc_idx=1, worker_model=worker_model, tagging_scheme='IOB2',
                data_model=data_model, transition_model='HMM')

    model.verbose = True

    # E_t = np.zeros((gt.shape[0], L), dtype=float)
    # E_t[range(gt.shape[0]), gt] = 1
    #
    # alg.worker_model._expand_alpha0(alpha0, None, annos.shape[1], L)
    # alpha = np.copy(alpha0)
    # diff = np.inf
    # while diff > 0:
    #     oldalpha = alpha
    #     alpha = alg.worker_model._post_alpha(E_t, annos, alpha0, alpha, doc_start, L, before_doc_idx=1)  # Posterior Hyperparameters)
    #     diff = np.sum(np.abs(oldalpha - alpha))

    # above: training on gold shows whether the worker behaviour patterns are detectable from the data in perfect
    # conditions. below: training with real dataset shows whether we can detect the worker behaviour patterns in a
    # realistic setting when gold labels are not present.

    model.max_iter = 20

    if 'LSTM' in data_model:
        import lample_lstm_tagger.lstm_wrapper as lstm_wrapper
        dev_sentences, _, _ = lstm_wrapper.data_to_lstm_format(len(gt_dev), text_dev,
                                                               doc_start_dev, gt_dev)
    else:
        dev_sentences = None

    probs, agg, _ = model.run(annos, doc_start, text, converge_workers_first=2 if 'LSTM' in data_model else 0,
                              gold_labels=gt)

    print('completed training with worker model = %s and data models %s' % (worker_model, data_model))

    # DATA MODELS
    for d, dm in enumerate(model.data_model):
        alpha_data = dm.alpha_data
        EPi = model.A._calc_EPi(alpha_data)

        # get the confusion matrices from the worker model
        if worker_model == 'acc':
            EPi_square = np.zeros((L, L))
            EPi_square[:, :] = EPi[0, 0] / (L - 1)
            EPi_square[range(L), range(L)] = EPi[1, 0]

        elif worker_model == 'vec' or worker_model == 'ibcc':
            EPi_square = EPi[:, :, 0]

        elif worker_model == 'seq':
            EPi_square = []  # create a separate conf mat for each previous label value
            for l in range(L):
                EPi_square.append(EPi[:, :, l, 0])

        elif worker_model == 'mace':

            EPi_square = np.zeros((L, L))
            # add the spamming strategy
            EPi_square[:, :] = (EPi[0, 0] * EPi[2:, 0])
            # add the probability of being correct
            EPi_square[range(L), range(L)] += EPi[1, 0]

        # Add a list of conf matrices to the dictionary, i.e. one per cluster,
        # except we do not cluster the data models here but plot them separately.
        if worker_model == 'seq':
            for i, EPi_square_i in enumerate(EPi_square):
                confmats[worker_model + '+' + data_model[d] + ', prev label = %i' % i] = [EPi_square_i]
        else:
            confmats[worker_model + '+' + data_model[d]] = [EPi_square]

    # WORKERS
    EPi_list = []
    EPi = model.A._calc_EPi(model.alpha)

    # get the confusion matrices from the worker model
    if worker_model == 'acc':
        EPi_square = np.zeros((L, L, annos.shape[1]))
        EPi_square[:, :, :] = EPi[0, :][None, None, :] / (L-1)
        EPi_square[range(L), range(L), :] = EPi[1, :][None, :]

        EPi_list.append(EPi_square)

    elif worker_model == 'vec' or worker_model == 'ibcc':
        EPi_list.append(EPi)

    elif worker_model == 'seq':
        EPi_square = [] # create a separate conf mat for each previous label value
        for l in range(L):
            EPi_square.append(EPi[:, :, l, :])

        EPi_list.append(EPi_square)

    elif worker_model == 'mace':

        EPi_square = np.zeros((L, L, annos.shape[1]))
        # add the spamming strategy
        EPi_square[:, :, :] = (EPi[0, :] * EPi[2:, :])[None, :, :]
        # add the probability of being correct
        EPi_square[range(L), range(L), :] += EPi[1, :][None, :]

        EPi_list.append(EPi_square)

    with open(output_dir + '/EPi_list_%s.pkl' % worker_model, 'w') as fh:
        pickle.dump(EPi_list, fh)

    for EPi in EPi_list:
        # cluster

        nclusters = 5

        EPi = np.array(EPi)
        if EPi.shape[-1] < nclusters:
            nclusters = EPi.shape[-1]

        EPi_vec = np.swapaxes(EPi, 0, -1).reshape(EPi.shape[-1], int(EPi.size / EPi.shape[-1]))
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=nclusters, batch_size=100,
                              n_init=50, max_no_improvement=10, verbose=0,
                              random_state=0)

        clusters = mbk.fit_predict(EPi_vec)

        nrows = 1
        if EPi.ndim > 3:
            nrows = EPi.shape[0]

        for i in range(nrows):
            mean_conf_mats = []

            if EPi.ndim > 3:
                EPi_i = EPi[i]
            else:
                EPi_i = EPi

            for c in range(nclusters):
                # compute mean confusion matrix
                if np.any(clusters == c):
                    mean_conf_mat = np.mean(EPi_i[:, :, clusters == c], axis=2)
                    mean_conf_mats.append(mean_conf_mat)

            if nrows > 1:
                confmats[worker_model + ', prev. label=%i' % i] = mean_conf_mats
            else:
                confmats[worker_model] = mean_conf_mats

    with open(output_dir + '/confmat_clustered_%s.pkl' % worker_model, 'w') as fh:
        pickle.dump(confmats, fh)

# Plot the confusion matrices in confmats

cols = [cm.hot(x/float(L)) for x in range(L)]#['b', 'g', 'r', 'y']
labels = ['I', 'O', 'B']

for wm in confmats:
    cmwm = confmats[wm]

    nsubfigs = len(cmwm)

    f = plt.figure(figsize=(nsubfigs * 3, 3))

    for s in range(nsubfigs):

        ax = plt.subplot(1, nsubfigs, s + 1, projection='3d')
        barslist = []

        for l in range(L):

            # x = np.arange(L) / float(L+1)
            x = np.zeros(L) + l - 0.5
            y = np.arange(L) - 0.5

            width = 0.8 / float(L)
            top = cmwm[s][l, :]

            bars = ax.bar3d(x, y, 0, 0.8, 0.8, top, alpha=0.8, edgecolor='k', zorder=l+100)
            bars._sort_zpos = l
            bars.set_alpha(0.8)

            barslist.append(bars)

            # ax.bar(x + l - 0.25, top, width, edgecolor='k', color=cols, alpha=0.9)
            #
            # for m in range(L):
            #     ax.text(x[m] + l - 0.33, 0.1 + np.min(top), '%i' % m, color='k' if np.sum(cols[m][:3]) > 0.5 or
            #                                                                        0.1 + np.min(top) > top[m] else 'w')

        ax.set_title(wm + ' cluster %i' % (s))

        ax.set_xticks(range(L))
        ax.set_yticks(range(L))

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.tick_params(axis='x', pad=-5)
        ax.tick_params(axis='y', pad=-5)
        ax.tick_params(axis='z', pad=-2)

        ax.set_xlabel('true label', labelpad=-5)
        ax.set_ylabel('worker label', labelpad=-5)

        if s == nsubfigs -1 :
            ax.set_zlabel('p(worker label | true label)', labelpad=-3)

        ax.set_xlim(-0.5, L-0.5)
        ax.set_ylim(-0.5, L-0.5)
        ax.set_zlim(0, 1)

    plt.tight_layout()
    #plt.show()
    for i, bars in enumerate(barslist):
        bars.set_alpha(0.8)

    # save plots to file...

    savepath = './documents/figures/worker_models/'
    plt.figure(f.number)
    plt.savefig(savepath + wm.replace(', prev. label=', '_prev') + '.pdf')
