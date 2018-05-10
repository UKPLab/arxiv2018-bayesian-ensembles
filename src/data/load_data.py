'''
Created on Jul 6, 2017

@author: Melvin Laux
'''

import os
import re
import sys

import pandas as pd
import numpy as np
import glob

from pandas._libs.parsers import na_values
from sklearn.feature_extraction.text import CountVectorizer

def convert_argmin(x):
    label = x.split('-')[0]
    if label == 'I':
        return 0
    if label == 'O':
        return 1
    if label == 'B':
        return 2

def convert_7class_argmin(x):
    label = x.split('-')[0]
    if label == 'I':
        label = x.split('-')[1].split(':')[0]
        if label == 'MajorClaim':
            return 0
        elif label == 'Claim':
            return 3
        elif label == 'Premise':
            return 5
    if label == 'O':
        return 1
    if label == 'B':
        label = x.split('-')[1].split(':')[0]
        if label == 'MajorClaim':        
            return 2
        elif label == 'Claim':
            return 4
        elif label == 'Premise':
            return 6
    
def convert_crowdsourcing(x):
    if x == 'Premise-I':
        return 0
    elif x == 'O':
        return 1
    elif x== 'Premise-B':
        return 2
    else:
        return -1
    

def load_argmin_data():    

    path = '../../data/bayesian_annotator_combination/data/argmin/'
    if not os.path.isdir(path):
        os.mkdir(path)
            
    all_files = glob.glob(os.path.join(path, "*.dat.out"))
    df_from_each_file = (pd.read_csv(f, sep='\t', usecols=(0, 5, 6), converters={5:convert_argmin, 6:convert_argmin},
                                     header=None, quoting=3) for f in all_files)
    concatenated = pd.concat(df_from_each_file, ignore_index=True, axis=1).as_matrix()

    annos = concatenated[:, 1::3]
    
    for t in range(1, annos.shape[0]):
        annos[t, (annos[t-1, :] == 1) & (annos[t, :] == 0)] = 2
    
    gt = concatenated[:, 2][:, None]
    doc_start = np.zeros((annos.shape[0], 1))    
    doc_start[np.where(concatenated[:, 0] == 1)] = 1

    # correct the base classifiers
    non_start_labels = [0]
    start_labels = [2] # values to change invalid I tokens to
    for l, label in enumerate(non_start_labels):
        start_annos = annos[doc_start.astype(bool).flatten(), :]
        start_annos[start_annos == label] = start_labels[l]
        annos[doc_start.astype(bool).flatten(), :] = start_annos

    np.savetxt('../../data/bayesian_annotator_combination/data/argmin/annos.csv', annos, fmt='%s', delimiter=',')
    np.savetxt('../../data/bayesian_annotator_combination/data/argmin/gt.csv', gt, fmt='%s', delimiter=',')
    np.savetxt('../../data/bayesian_annotator_combination/data/argmin/doc_start.csv', doc_start, fmt='%s', delimiter=',')

    return gt, annos, doc_start

def load_argmin_7class_data():    

    path = '../../data/bayesian_annotator_combination/data/argmin/'
    if not os.path.isdir(path):
        os.mkdir(path)
    
    all_files = glob.glob(os.path.join(path, "*.dat.out"))
    df_from_each_file = (pd.read_csv(f, sep='\t', usecols=(0, 5, 6), converters={5:convert_7class_argmin,
                                                    6:convert_7class_argmin}, header=None, quoting=3) for f in all_files)
    concatenated = pd.concat(df_from_each_file, ignore_index=True, axis=1).as_matrix()

    annos = concatenated[:, 1::3]
    gt = concatenated[:, 2][:, None]
    doc_start = np.zeros((annos.shape[0], 1))    
    doc_start[np.where(concatenated[:, 0] == 1)] = 1

    # correct the base classifiers
    non_start_labels = [0, 3, 5]
    start_labels = [2, 4, 6] # values to change invalid I tokens to
    for l, label in enumerate(non_start_labels):
        start_annos = annos[doc_start.astype(bool).flatten(), :]
        start_annos[start_annos == label] = start_labels[l]
        annos[doc_start.astype(bool).flatten(), :] = start_annos

    outpath = '../../data/bayesian_annotator_combination/data/argmin7/'
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    
    np.savetxt(outpath + 'annos.csv', annos, fmt='%s', delimiter=',')
    np.savetxt(outpath + 'gt.csv', gt, fmt='%s', delimiter=',')
    np.savetxt(outpath + 'doc_start.csv', doc_start, fmt='%s', delimiter=',')

    return gt, annos, doc_start

def load_crowdsourcing_data():
    path = '../../data/bayesian_annotator_combination/data/crowdsourcing/'
    if not os.path.isdir(path):
        os.mkdir(path)
            
    all_files = glob.glob(os.path.join(path, "exported*.csv"))
    print(all_files)
    
    convs = {}
    for i in range(1,50):
        convs[i] = convert_crowdsourcing
    
    df_from_each_file = [pd.read_csv(f, sep=',', header=None, skiprows=1, converters=convs) for f in all_files]
    concatenated = pd.concat(df_from_each_file, ignore_index=False, axis=1).as_matrix()
    
    concatenated = np.delete(concatenated, 25, 1);
    
    annos = concatenated[:,1:]
    
    doc_start = np.zeros((annos.shape[0],1))
    doc_start[0] = 1    
    
    for i in range(1,annos.shape[0]):
        if '_00' in str(concatenated[i,0]):
            doc_start[i] = 1
    
    np.savetxt('../../data/bayesian_annotator_combination/data/crowdsourcing/gen/annos.csv', annos, fmt='%s', delimiter=',')
    np.savetxt('../../data/bayesian_annotator_combination/data/crowdsourcing/gen/doc_start.csv', doc_start, fmt='%s', delimiter=',')
    
    return annos, doc_start

def build_feature_vectors(text_data_arr):
    text_data_arr = np.array(text_data_arr).astype(str)

    vectorizer = CountVectorizer()
    count_vectors = vectorizer.fit_transform(text_data_arr) # each element can be a sentence or a single word
    count_vectors = count_vectors.toarray() # each row will be a sentence
    return count_vectors, vectorizer.get_feature_names()

def _load_pico_feature_vectors_from_file(corpus):

    all_text = []
    for docid in corpus.docs:
        text_d = corpus.get_doc_text(docid)
        all_text.append(text_d)

    feature_vecs, _ = build_feature_vectors(all_text)

    return feature_vecs

def _load_bio_folder(anno_path_root, folder_name):
    '''
    Loads one data directory out of the complete collection.
    :return: dataframe containing the data from this folder.
    '''
    from data.pico.corpus import Corpus

    DOC_PATH = os.path.expanduser("~/git/PICO-data/docs/")
    ANNOTYPE = 'Participants'

    anno_path = anno_path_root + folder_name

    anno_fn = anno_path + '/PICO-annos-crowdsourcing.json'
    gt_fn = anno_path + '/PICO-annos-professional.json'

    corpus = Corpus(doc_path=DOC_PATH, verbose=False)

    corpus.load_annotations(anno_fn, docids=None)
    if os.path.exists(gt_fn):
        corpus.load_groundtruth(gt_fn)

    # get a list of the docids
    docids = []

    workerids = np.array([], dtype=str)

    all_data = None

    #all_fv = _load_pico_feature_vectors_from_file(corpus)

    for d, docid in enumerate(corpus.docs):
        docids.append(docid)

        annos_d = corpus.get_doc_annos(docid, ANNOTYPE)

        spacydoc = corpus.get_doc_spacydoc(docid)
        text_d = spacydoc #all_fv[d]
        doc_length = len(text_d)

        doc_data = None

        for workerid in annos_d:

            print('Processing data for doc %s and worker %s' % (docid, workerid))

            if workerid not in workerids:
                workerids = np.append(workerids, workerid)

            # add the worker to the dataframe if not already there
            if doc_data is None or workerid not in doc_data:
                doc_data_w = np.ones(doc_length, dtype=int)  # O tokens

                if doc_data is None:
                    doc_data = pd.DataFrame(doc_data_w, columns=[workerid])
                else:
                    doc_data[workerid] = doc_data_w
            else:
                doc_data_w = doc_data[workerid]

            for span in annos_d[workerid]:
                start = span[0]
                fin = span[1]

                doc_data_w[start] = 2
                doc_data_w[start + 1:fin] = 0

        if os.path.exists(gt_fn):

            gold_d = corpus.get_doc_groundtruth(docid, ANNOTYPE)

            if 'gold' not in doc_data:
                doc_data['gold'] = np.ones(doc_length, dtype=int)

            for spans in gold_d:
                start = spans[0]
                fin = spans[1]

                doc_data['gold'][start] = 2
                doc_data['gold'][start + 1:fin] = 0
        else:
            doc_data['gold'] = np.zeros(doc_length, dtype=int) - 1  # -1 for missing gold values

        doc_data['doc_start'] = np.zeros(doc_length, dtype=int)
        doc_data['doc_start'][0] = 1

        text_d = [spacytoken.text for spacytoken in text_d]
        doc_data['text'] = text_d
        doc_data = doc_data.replace(r'\n', ' ', regex=True)

        doc_data['docid'] = docid

        if all_data is None:
            all_data = doc_data
        else:
            all_data = pd.concat([all_data, doc_data])

        # print('breaking for fast debugging')
        # break

    return all_data, workerids

def load_biomedical_data(regen_data_files):
    savepath = '../../data/bayesian_annotator_combination/data/bio/'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    if regen_data_files:
        anno_path_root = '../../data/bayesian_annotator_combination/data/bio-PICO/'

        # There are four folders here:
        # acl17-test: the only one containing 'professional' annotations. 191 docs
        # train: 3549 docs
        # dev: 500 docs
        # test: 500 docs

        # Total of 4740 is slightly fewer than the values stated in the paper.
        # The validation/test split in the acl17-test data is also not given. This suggests we may need to run the
        # HMMCrowd and LSTMCrowd methods with hyperparameter tuning on our own splits. Let's skip that tuning for now?
        # Cite Nils' paper about using a generic hyperparameter tuning that works well across tasks -- we need to do
        # this initially because we don't have gold data to optimise on.
        # Nguyen et al do only light tuning with a few (less than 5) values to choose from for each hyperparameter of
        # HMM-Crowd, LSTM-Crowd and the individual LSTMs. Not clear whether the values set in the code as default are
        # the chosen values -- let's assume so for now. We can re-tune later if necessary. Remember: we don't require
        # a validation set for tuning our methods.

        # We need for task1 and task2:
        # train, dev and test splits.
        # I believe the acl17-test set was split to form the dev and test sets in nguyen et al.
        # Task 1 does not require separate training samples -- it's trained on crowdsourced rather than professional labels.
        # Task 2 requires testing on separate samples (with gold labels)
        # from the training samples (with crowd labels).
        # Both tasks use all training data for training and the acl17-test set for validation/testing.
        # These other splits into the train, test and dev folders appear to relate to a different set of experiments
        # and are not relevant to nguyen et al 2017.

        folders_to_load = ['acl17-test', 'train', 'test', 'dev']
        all_data = None
        all_workerids = None

        for folder in folders_to_load:
            folder_data, workerids = _load_bio_folder(anno_path_root, folder)

            if all_data is None:
                all_data = folder_data
                all_workerids = workerids
            else:
                all_data = pd.concat([all_data, folder_data])
                all_workerids = np.unique(np.append(workerids.flatten(), all_workerids.flatten()))

            all_data.to_csv(savepath + '/annos.csv', columns=all_workerids, header=False, index=False)
            all_data.to_csv(savepath + '/gt.csv', columns=['gold'], header=False, index=False)
            all_data.to_csv(savepath + '/doc_start.csv', columns=['doc_start'], header=False, index=False)
            all_data.to_csv(savepath + '/text.csv', columns=['text'], header=False, index=False)

    print('loading annos...')
    annos = pd.read_csv(savepath + '/annos.csv', header=None)
    annos = annos.fillna(-1)
    annos = annos.values
    #np.genfromtxt(savepath + '/annos.csv', delimiter=',')

    print('loading text data...')
    text = pd.read_csv(savepath + './text.csv', skip_blank_lines=False, header=None).values

    print('loading doc starts...')
    doc_start = pd.read_csv(savepath + '/doc_start.csv', header=None).values #np.genfromtxt(savepath + '/doc_start.csv')

    print('loading ground truth labels...')
    gt = pd.read_csv(savepath + '/gt.csv', header=None).values # np.genfromtxt(savepath + '/gt.csv')

    if len(text) == len(annos) - 1:
        # sometimes the last line of text is blank and doesn't get loaded into text, but doc_start and gt contain labels
        # for the newline token
        annos = annos[:-1]
        doc_start = doc_start[:-1]
        gt = gt[:-1]

    print('Creating dev/test split...')
    # since there is no separate validation set, we split the test set
    ndocs = len(gt)
    testidxs = np.random.randint(0, ndocs, int(np.floor(ndocs * 0.5)))

    devidxs = np.ones(ndocs, dtype=bool)
    devidxs[testidxs] = False

    gt_test = np.copy(gt)
    gt_test[devidxs] = -1

    gt_dev = gt[devidxs]
    doc_start_dev = doc_start[devidxs]
    text_dev = text[devidxs]

    return gt_test, annos, doc_start, text, gt_dev, doc_start_dev, text_dev

def _map_ner_str_to_labels(arr):

    arr = arr.astype(str)

    arr[arr == 'O'] = 1
    arr[arr == 'B-ORG'] = 2
    arr[arr == 'I-ORG'] = 0
    arr[arr == 'B-PER'] = 4
    arr[arr == 'I-PER'] = 3
    arr[arr == 'B-LOC'] = 6
    arr[arr == 'I-LOC'] = 5
    arr[arr == 'B-MISC'] = 8
    arr[arr == 'I-MISC'] = 7

    try:
        arr_ints = arr.astype(int)
    except:
        print("Could not map all annotations to integers. The annotations we found were:")
        print(np.unique(arr))

    return arr_ints


def _load_rodrigues_annotations(dir, worker_str):
    worker_data = None

    for f in os.listdir(dir):

        if not f.endswith('.txt'):
            continue

        doc_str = f.split('.')[0]
        f = os.path.join(dir, f)

        print('Processing %s' % f)

        new_data = pd.read_csv(f, delimiter=' ', names=['text', worker_str], skip_blank_lines=True,
                               dtype={'text':str, worker_str:str}, na_filter=False, sep=' ')

        new_data[worker_str] = _map_ner_str_to_labels(new_data[worker_str])

        new_data['doc_id'] = doc_str

        new_data['tok_idx'] = new_data.index

        new_data['doc_start'] = (new_data.index == 0).astype(int)

        # add to data from this worker
        if worker_data is None:
            worker_data = new_data
        else:
            worker_data = pd.concat([worker_data, new_data])

    return worker_data

def _load_rodrigues_annotations_all_workers(annotation_data_path):

    worker_dirs = os.listdir(annotation_data_path)

    data = None
    annotator_cols = np.array([], dtype=str)

    for widx, dir in enumerate(worker_dirs):

        if dir.startswith("."):
            continue

        worker_str = dir
        annotator_cols = np.append(annotator_cols, worker_str)
        dir = os.path.join(annotation_data_path, dir)

        print('Processing dir for worker %s (%i of %i)' % (worker_str, widx, len(worker_dirs)))

        worker_data = _load_rodrigues_annotations(dir, worker_str)

        # now need to join this to other workers' data
        if data is None:
            data = worker_data
        else:
            data = data.merge(worker_data, on=['doc_id', 'tok_idx', 'text', 'doc_start'], how='outer')

    return data, annotator_cols

def load_ner_data(regen_data_files):
    # In Nguyen et al 2017, the original data has been separated out for task 1, aggregation of crowd labels. In this
    # task, the original training data is further split into val and test -- to make our results comparable with Nguyen
    # et al, we need to test on the test split for task 1, but train our model on both.
    # To make them comparable with Rodrigues et al. 2014, we need to test on all data (check this in their paper).
    # Task 2 is for prediction on a test set given a model trained on the training set and optimised on the validation
    # set. It would be ideal to show both these results...

    savepath = '../../data/bayesian_annotator_combination/data/ner/' # location to save our csv files to
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    # within each of these folders below is an mturk_train_data folder, containing crowd labels, and a ground_truth
    # folder. Rodrigues et al. have assigned document IDs that allow us to match up the annotations from each worker.
    # Nguyen et al. have split the training set into the val/test folders for task 1. Data is otherwise the same as in
    # the Rodrigues folder under mturk/extracted_data.
    task1_val_path = '../../data/bayesian_annotator_combination/data/crf-ma-NER-task1/val/'
    task1_test_path = '../../data/bayesian_annotator_combination/data/crf-ma-NER-task1/test/'

    # These are just two files that we use for text features + ground truth labels.
    task2_val_path = '../../data/bayesian_annotator_combination/data/English NER/eng.testa'
    task2_test_path = '../../data/bayesian_annotator_combination/data/English NER/eng.testa'

    if regen_data_files:
        # Steps to load data (all steps need to map annotations to consecutive integer labels).
        # 1. Create an annos.csv file containing all the annotations in task1_val_path and task1_test_path.
        # load the validation data
        data, annotator_cols = _load_rodrigues_annotations_all_workers(task1_val_path + 'mturk_train_data/')

        # convert the text to feature vectors
        #data['text'], _ = build_feature_vectors(data['text'].values)

        # save the annos.csv
        data.to_csv(savepath + '/task1_val_annos.csv', columns=annotator_cols, header=False, index=False,
                    float_format='%.f', na_rep=-1)

        # save the text in same order
        data.to_csv(savepath + '/task1_val_text.csv', columns=['text'], header=False, index=False)

        # save the doc starts
        data.to_csv(savepath + '/task1_val_doc_start.csv', columns=['doc_start'], header=False, index=False)


        # 2. Create ground truth CSV for task1_val_path (for tuning the LSTM)
        # load the gold data in the same way as the worker data
        gold_data = _load_rodrigues_annotations(task1_val_path + 'ground_truth/', 'gold')

        # merge with the worker data
        data = data.merge(gold_data, on=['doc_id', 'tok_idx'])

        # save the annos.csv
        data.to_csv(savepath + '/task1_val_gt.csv', columns=['gold'], header=False, index=False)


        # 3. Load worker annotations for test set.
        # load the test data
        data, annotator_cols = _load_rodrigues_annotations_all_workers(task1_test_path + 'mturk_train_data/')

        # convert the text to feature vectors
        data['text'], _ = build_feature_vectors(data['text'].values)

        # save the annos.csv
        data.to_csv(savepath + '/task1_test_annos.csv', columns=annotator_cols, header=False, index=False,
                    float_format='%.f', na_rep=0)

        # save the text in same order
        data.to_csv(savepath + '/task1_test_text.csv', columns=['text'], header=False, index=False)

        # save the doc starts
        data.to_csv(savepath + '/task1_test_doc_start.csv', columns=['doc_start'], header=False, index=False)


        # 4. Create ground truth CSV for task1_test_path
        # load the gold data in the same way as the worker data
        gold_data = _load_rodrigues_annotations(task1_test_path + 'ground_truth/', 'gold')

        # merge with the worker data
        data = data.merge(gold_data, on=['doc_id', 'tok_idx'])

        # save the annos.csv
        data.to_csv(savepath + '/task1_test_gt.csv', columns=['gold'], header=False, index=False)


        # 5. Create a file containing only the words for the task 2 validation set, i.e. like annos.csv with no annotations.
        # Create ground truth CSV for task1_val_path, task1_test_path and task2_val_path but blank out the task_1 labels
        # (for tuning the LSTM for task 2)

        eng_val = pd.read_csv(task2_val_path, delimiter=' ', usecols=[0,3], names=['text', 'gold'], skip_blank_lines=True)

        doc_starts = np.zeros(eng_val.shape[0])
        docstart_token = eng_val['text'][0]

        doc_starts[1:] = (eng_val['text'] == docstart_token)[:-1]
        eng_val['doc_start'] = doc_starts

        eng_val['tok_idx'] = eng_val.index
        eng_val = eng_val[eng_val['text'] != docstart_token] # remove all the docstart labels

        eng_val.to_csv(savepath + '/task2_val_text.csv', columns=['gold'], header=False, index=False)
        eng_val.to_csv(savepath + '/task2_val_gt.csv', columns=['text'], header=False, index=False)
        eng_val.to_csv(savepath + '/task2_val_doc_start.csv', columns=['doc_start'], header=False, index=False)


        # 6. Create a file containing only the words for the task 2 test set, i.e. like annos.csv with no annotations.
        # Create ground truth CSV for task1_val_path, task1_test_path and task2_test_path but blank out the task_1 labels/
        eng_val = pd.read_csv(task2_test_path, delimiter=' ', usecols=[0,3], names=['text', 'gold'], skip_blank_lines=True)

        doc_starts = np.zeros(eng_val.shape[0])
        docstart_token = eng_val['text'][0]

        doc_starts[1:] = (eng_val['text'] == docstart_token)[:-1]
        eng_val['doc_start'] = doc_starts

        eng_val['tok_idx'] = eng_val.index
        eng_val = eng_val[eng_val['text'] != docstart_token] # remove all the docstart labels

        eng_val.to_csv(savepath + '/task2_test_text.csv', columns=['gold'], header=False, index=False)
        eng_val.to_csv(savepath + '/task2_test_gt.csv', columns=['text'], header=False, index=False)
        eng_val.to_csv(savepath + '/task2_test_doc_start.csv', columns=['doc_start'], header=False, index=False)


    # 7. Reload the data for the current run...
    print('loading annos for task1 test...')
    annos = np.genfromtxt(savepath + '/task1_test_annos.csv', delimiter=',')

    print('loading text data for task1 test...')
    text = np.genfromtxt(savepath + '/task1_test_text.csv')

    print('loading doc_starts for task1 test...')
    doc_start = np.genfromtxt(savepath + '/task1_test_doc_start.csv')

    print('loading ground truth for task1 test...')
    gt = np.genfromtxt(savepath + '/task1_test_gt.csv')

    print('loading annos for task1 val...')
    annos_v = np.genfromtxt(savepath + '/task1_val_annos.csv', delimiter=',')
    annos = np.concatenate((annos, annos_v), axis=0)

    print('loading text data for task1 val...')
    text_v = np.genfromtxt(savepath + '/task1_test_text.csv')
    text = np.concatenate((text, text_v), axis=0)

    print('loading doc_starts for task1 val...')
    doc_start_v = np.genfromtxt(savepath + '/task1_val_doc_start.csv')
    doc_start = np.concatenate((doc_start, doc_start_v), axis=0)

    print('not loading ground truth for task1 val, as we have done all hyperparameter tuning before...')
    gt_v = np.zeros(annos_v.shape[0]) - 1
    gt = np.concatenate((gt, gt_v), axis=0)

    print('loading text data for task 2 test')
    text_task2 = np.genfromtxt(savepath + '/task2_test_text.csv')

    print('loading doc_starts for task 2 test')
    doc_start_task2 = np.genfromtxt(savepath + '/task2_test_doc_start.csv')

    print('loading ground truth for task 2 test')
    gt_task2 = np.genfromtxt(savepath + '/task2_test_gt.csv')

    # validation sets for methods that predict on features only
    print('loading text data for task 2 val')
    text_val = np.genfromtxt(savepath + '/task2_val_text.csv')

    print('loading doc_starts for task 2 val')
    doc_start_val = np.genfromtxt(savepath + '/task2_val_doc_start.csv')

    print('loading ground truth for task 2 val')
    gt_val = np.genfromtxt(savepath + '/task2_val_gt.csv')

    return gt, annos, doc_start, text, gt_task2, doc_start_task2, text_task2, gt_val, doc_start_val, text_val

if __name__ == '__main__':
    pass
