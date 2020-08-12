'''

Scripts for loading various experimental datasets.

Created on Jul 6, 2017

@author: Melvin Laux
'''

import os
import pandas as pd
import numpy as np
from evaluation.experiment import data_root_dir

all_root_dir = data_root_dir#os.path.expanduser('~/data/bayesian_sequence_combination')
data_root_dir = os.path.join(all_root_dir, 'data')
output_root_dir = os.path.join(all_root_dir, 'output')


def _load_bio_folder(anno_path_root, folder_name):
    '''
    Loads one data directory out of the complete collection.
    :return: dataframe containing the data from this folder.
    '''
    from data.pico.corpus import Corpus

    DOC_PATH = os.path.join(data_root_dir, "bio-PICO/docs/")
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
                doc_data_w = doc_data[workerid]

            for span in annos_d[workerid]:
                start = span[0]
                fin = span[1]

                doc_data_w[start] = 2
                doc_data_w[start + 1:fin] = 0

            doc_data[workerid] = doc_data_w

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

        text_d = [spacytoken.text for spacytoken in text_d]
        doc_data['text'] = text_d

        doc_start = np.zeros(doc_length, dtype=int)
        doc_start[0] = 1

        doc_gaps = doc_data['text'] == '\n\n' # sentence breaks
        doc_start[doc_gaps[doc_gaps].index[:-1] + 1] = 1
        doc_data['doc_start'] = doc_start

        # doc_data = doc_data.replace(r'\n', ' ', regex=True)

        doc_data = doc_data[np.invert(doc_gaps)]

        doc_data['docid'] = docid

        if all_data is None:
            all_data = doc_data
        else:
            all_data = pd.concat([all_data, doc_data], axis=0)

        # print('breaking for fast debugging')
        # break

    return all_data, workerids

def load_biomedical_data(regen_data_files, debug_subset_size=None, data_folder='bio'):
    savepath = os.path.join(data_root_dir, data_folder)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    if regen_data_files or not os.path.isfile(savepath + '/annos.csv'):

        print(regen_data_files)
        print(os.path.isfile(savepath + '/annos.csv'))

        anno_path_root = os.path.join(data_root_dir, 'bio-PICO/annos/')

        # There are four folders here:
        # acl17-test: the only one containing 'professional' annos. 191 docs
        # train: 3549 docs
        # dev: 500 docs
        # test: 500 docs

        folders_to_load = ['acl17-test', 'train', 'test', 'dev']
        all_data = None
        all_workerids = None

        for folder in folders_to_load:
            print('Loading folder %s' % folder)
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
    annos = pd.read_csv(savepath + '/annos.csv', header=None, nrows=debug_subset_size)
    annos = annos.fillna(-1)
    annos = annos.values
    #np.genfromtxt(savepath + '/annos.csv', delimiter=',')

    print('loading text data...')
    text = pd.read_csv(savepath + '/text.csv', skip_blank_lines=False, header=None, nrows=debug_subset_size)
    text = text.fillna(' ').values

    print('loading doc starts...')
    doc_start = pd.read_csv(savepath + '/doc_start.csv', header=None, nrows=debug_subset_size).values #np.genfromtxt(savepath + '/doc_start.csv')
    print('Loaded %i documents' % np.sum(doc_start))

    print('loading ground truth labels...')
    gt = pd.read_csv(savepath + '/gt.csv', header=None, nrows=debug_subset_size).values # np.genfromtxt(savepath + '/gt.csv')

    if len(text) == len(annos) - 1:
        # sometimes the last line of text is blank and doesn't get loaded into text, but doc_start and gt contain labels
        # for the newline token
        annos = annos[:-1]
        doc_start = doc_start[:-1]
        gt = gt[:-1]

    print('Creating dev/test split...')

    # since there is no separate validation set, we split the test set
    ndocs = np.sum(doc_start & (gt != -1))
    #testdocs = np.random.randint(0, ndocs, int(np.floor(ndocs * 0.5)))
    ntestdocs = int(np.floor(ndocs * 0.5))
    docidxs = np.cumsum(doc_start & (gt != -1)) # gets us the doc ids
    # # testidxs = np.in1d(docidxs, testdocs)
    ntestidxs = np.argwhere(docidxs == (ntestdocs+1))[0][0]

    # The first half of the labelled data is used as dev, second half as test
    gt_test = np.copy(gt)
    gt_test[ntestidxs:] = -1

    gt_dev = np.copy(gt)
    gt_dev[:ntestidxs] = -1

    doc_start_dev = doc_start[gt_dev != -1]
    text_dev = text[gt_dev != -1]

    gt_task1_dev = gt_dev

    gt_dev = gt_dev[gt_dev != -1]

    return gt_test, annos, doc_start, text, gt_task1_dev, gt_dev, doc_start_dev, text_dev


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
    arr[arr == '?'] = -1

    try:
        arr_ints = arr.astype(int)
    except:
        print("Could not map all annos to integers. The annos we found were:")
        uannos = []
        for anno in arr:
            if anno not in uannos:
                uannos.append(anno)
        print(uannos)

    return arr_ints


def _load_rodrigues_annotations(dir, worker_str, gold_char_idxs=None, gold_tokens=None, skip_imperfect_matches=False):
    worker_data = None

    for f in os.listdir(dir):

        if not f.endswith('.txt'):
            continue

        doc_str = f.split('.')[0]
        f = os.path.join(dir, f)

        #print('Processing %s' % f)

        new_data = pd.read_csv(f, names=['text', worker_str], skip_blank_lines=False,
                               dtype={'text':str, worker_str:str}, na_filter=False, delim_whitespace=True)

        doc_gaps = (new_data['text'] == '') & (new_data[worker_str] == '')
        doc_start = np.zeros(doc_gaps.shape[0], dtype=int)
        doc_start[doc_gaps[:-1][doc_gaps[:-1]].index + 1] = 1 # the indexes after the gaps
        doc_content = new_data['text'] != ''

        new_data['doc_start'] = doc_start
        new_data = new_data[doc_content]
        new_data['doc_start'].iat[0] = 1

        annos_to_keep = np.ones(new_data.shape[0], dtype=bool)

        for t, tok in enumerate(new_data['text']):

            if len(tok.split('/')) > 1:
                tok = tok.split('/')[0]
                new_data['text'].iat[t] = tok

            if len(tok) == 0:
                annos_to_keep[t] = False


        # compare the tokens in the worker annos to the gold labels. They are misaligned in the dataset. We will
        # skip labels in the worker annos that are assigned to only a part of a token in the gold dataset.
        char_counter = 0
        gold_tok_idx = 0

        skip_sentence = False
        sentence_start = 0

        if gold_char_idxs is not None:

            gold_chars = np.array(gold_char_idxs[doc_str])

            last_accepted_tok = ''
            last_accepted_idx = -1

            for t, tok in enumerate(new_data['text']):

                if skip_imperfect_matches and skip_sentence:
                    new_data[worker_str].iloc[t] = -1

                    if new_data['doc_start'].iat[t]:
                        skip_sentence = False

                if new_data['doc_start'].iat[t]:
                    sentence_start = t

                gold_char_idx = gold_chars[gold_tok_idx]
                gold_tok = gold_tokens[doc_str][gold_tok_idx]

                #print('tok = %s, gold_tok = %s' % (tok, gold_tok))

                if not annos_to_keep[t]:
                    continue # already marked as skippable

                if char_counter < gold_char_idx and \
                        (last_accepted_tok + tok) in gold_tokens[doc_str][gold_tok_idx-1]:
                    print('Correcting misaligned annos (split word in worker data): %i, %s' % (t, tok))

                    skip_sentence = True

                    last_accepted_tok += tok

                    annos_to_keep[last_accepted_idx] = False  # skip the previous ones until the end

                    new_data['text'].iat[t] = last_accepted_tok
                    new_data['doc_start'].iat[t] = new_data['doc_start'].iat[last_accepted_idx]

                    last_accepted_idx = t

                    char_counter += len(tok)

                elif tok not in gold_tok or (tok == '' and gold_tok != ''):
                    print('Correcting misaligned annos (spurious text in worker data): %i, %s vs. %s' % (t, tok, gold_tok))

                    skip_sentence = True

                    annos_to_keep[t] = False # skip the previous ones until the end

                    if new_data['doc_start'].iat[t]: # now we are skipping this token but we don't want to lose the doc_start record.
                        new_data['doc_start'].iat[t+1] = 1

                elif tok == gold_tok[:len(tok)]: # needs to match the first characters in the string, not just be there somewhere
                    gold_tok_idx += 1

                    if tok != gold_tok:
                        skip_sentence = True

                    while char_counter > gold_char_idx:
                        print('error in text alignment between worker and gold!')
                        len_to_skip = gold_chars[gold_tok_idx - 1] - gold_chars[gold_tok_idx - 2]

                        # move the gold counter along to the next token because gold is behind
                        gold_tok_idx += 1

                        gold_chars[gold_tok_idx:] -= len_to_skip

                        gold_char_idx = gold_chars[gold_tok_idx]

                    gold_char_idxs[doc_str] = gold_chars

                    last_accepted_tok = tok
                    last_accepted_idx = t

                    char_counter += len(tok)
                else:
                    skip_sentence = True

                    annos_to_keep[t] = False

                    if new_data['doc_start'].iat[t]: # now we are skipping this token but we don't want to lose the doc_start record.
                        new_data['doc_start'].iat[t+1] = 1

        # no more text in this document, but the last sentence must be skipped
        if skip_imperfect_matches and skip_sentence:
            # annos_to_keep[sentence_start:t+1] = False
            new_data[worker_str].iloc[sentence_start:t+1] = -1

        new_data = new_data[annos_to_keep]

        new_data[worker_str] = _map_ner_str_to_labels(new_data[worker_str])

        new_data['doc_id'] = doc_str

        new_data['tok_idx'] = np.arange(new_data.shape[0])

        # add to data from this worker
        if worker_data is None:
            worker_data = new_data
        else:
            worker_data = pd.concat([worker_data, new_data])

    return worker_data

def _load_rodrigues_annotations_all_workers(annotation_data_path, gold_data, skip_dirty=False):

    worker_dirs = os.listdir(annotation_data_path)

    data = None
    annotator_cols = np.array([], dtype=str)

    char_idx_word_starts = {}
    chars = {}

    char_counter = 0
    for t, tok in enumerate(gold_data['text']):
        if gold_data['doc_id'].iloc[t] not in char_idx_word_starts:
            char_counter = 0
            starts = []
            toks = []
            char_idx_word_starts[gold_data['doc_id'].iloc[t]] = starts

            chars[gold_data['doc_id'].iloc[t]] = toks

        starts.append(char_counter)
        toks.append(tok)

        char_counter += len(tok)

    for widx, dir in enumerate(worker_dirs):

        if dir.startswith("."):
            continue

        worker_str = dir
        annotator_cols = np.append(annotator_cols, worker_str)
        dir = os.path.join(annotation_data_path, dir)

        print('Processing dir for worker %s (%i of %i)' % (worker_str, widx, len(worker_dirs)))

        worker_data = _load_rodrigues_annotations(dir, worker_str,
                                                            char_idx_word_starts, chars, skip_dirty)

        print("Loaded a dataset of size %s" % str(worker_data.shape))

        # now need to join this to other workers' data
        if data is None:
            data = worker_data
        else:
            data = data.merge(worker_data, on=['doc_id', 'tok_idx', 'text', 'doc_start'], how='outer', sort=True, validate='1:1')

    return data, annotator_cols


def IOB_to_IOB2(seq):

    I_labels = [0, 3, 5, 7]
    B_labels = [2, 4, 6, 8]

    for i, label in enumerate(seq):
        if label in I_labels:

            typeidx = np.argwhere(I_labels == label)[0][0]

            if i == 0 or (seq[i-1] != B_labels[typeidx] and seq[i-1] != label):
                # we have I preceded by O. This needs to be changed to a B.
                seq[i] = B_labels[typeidx]

    return seq


def IOB2_to_IOB(seq):

    I_labels = [0, 3, 5, 7]
    B_labels = [2, 4, 6, 8]

    for i, label in enumerate(seq):
        if label in B_labels:

            typeidx = np.argwhere(B_labels == label)[0][0]

            if i == 0 or (seq[i-1] != B_labels[typeidx] or seq[i-1] != I_labels[typeidx]):
                # we have I preceded by O. This needs to be changed to a B.
                seq[i] = I_labels[typeidx]

    return seq


def load_ner_data(regen_data_files, skip_sen_with_dirty_data=False):
    # In Nguyen et al 2017, the original data has been separated out for task 1, aggregation of crowd labels. In this
    # task, the original training data is further split into val and test -- to make our results comparable with Nguyen
    # et al, we need to test on the test split for task 1, but train our model on both.
    # To make them comparable with Rodrigues et al. 2014, we need to test on all data (check this in their paper).
    # Task 2 is for prediction on a test set given a model trained on the training set and optimised on the validation
    # set. It would be ideal to show both these results...

    savepath = os.path.join(data_root_dir, 'ner') # location to save our csv files to
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    # within each of these folders below is an mturk_train_data folder, containing crowd labels, and a ground_truth
    # folder. Rodrigues et al. have assigned document IDs that allow us to match up the annos from each worker.
    # Nguyen et al. have split the training set into the val/test folders for task 1. Data is otherwise the same as in
    # the Rodrigues folder under mturk/extracted_data.
    task1_val_path = os.path.join(data_root_dir, 'crf-ma-NER-task1/val/')
    task1_test_path = os.path.join(data_root_dir, 'crf-ma-NER-task1/test')

    # These are just two files that we use for text features + ground truth labels.
    task2_val_path = os.path.join(data_root_dir, 'English NER/eng.testa')
    task2_test_path = os.path.join(data_root_dir, 'English NER/eng.testb')

    if regen_data_files or not os.path.isfile(savepath + '/task1_val_annos.csv'):
        # Steps to load data (all steps need to map annos to consecutive integer labels).
        # 1. Create an annos.csv file containing all the annos in task1_val_path and task1_test_path.

        # load the gold data in the same way as the worker data
        gold_data = _load_rodrigues_annotations(os.path.join(task1_val_path, 'ground_truth/'), 'gold')

        # load the validation data
        data, annotator_cols = _load_rodrigues_annotations_all_workers(
            os.path.join(task1_val_path, 'mturk_train_data/'),
            gold_data, skip_sen_with_dirty_data)

        # 2. Create ground truth CSV for task1_val_path (for tuning the LSTM)
        # merge gold with the worker data
        data = data.merge(gold_data, how='outer', on=['doc_id', 'tok_idx', 'doc_start', 'text'], sort=True)

        num_annotations = np.zeros(data.shape[0]) # count annos per token
        for col in annotator_cols:
            num_annotations += np.invert(data[col].isna())

        for doc in np.unique(data['doc_id']):
            # get tokens from this doc
            drows = data['doc_id'] == doc

            # get the annotation counts for this doc
            counts = num_annotations[drows]

            # check that all tokens have same number of annos
            if len(np.unique(counts)) > 1:
                print('Validation data: we have some misaligned labels.')
                print(counts)

            if np.any(counts.values == 0):
                print('Removing document %s with no annos.' % doc)

        # remove any lines with no annos
        annotated_idxs = num_annotations >= 1
        data = data[annotated_idxs]

        # save the annos.csv
        data.to_csv(savepath + '/task1_val_annos.csv', columns=annotator_cols, index=False,
                    float_format='%.f', na_rep=-1)

        # save the text in same order
        data.to_csv(savepath + '/task1_val_text.csv', columns=['text'], header=False, index=False)

        # save the doc starts
        data.to_csv(savepath + '/task1_val_doc_start.csv', columns=['doc_start'], header=False, index=False)

        # save the annos.csv
        data.to_csv(savepath + '/task1_val_gt.csv', columns=['gold'], header=False, index=False)

        # 3. Load worker annos for test set.
        # load the gold data in the same way as the worker data
        gold_data = _load_rodrigues_annotations(
            os.path.join(task1_test_path, 'ground_truth/'), 'gold')

        # load the test data
        data, annotator_cols = _load_rodrigues_annotations_all_workers(
            os.path.join(task1_test_path, 'mturk_train_data/'),
            gold_data, skip_sen_with_dirty_data)

        # 4. Create ground truth CSV for task1_test_path
        # merge with the worker data

        data = data.merge(gold_data, how='outer', on=['doc_id', 'tok_idx', 'doc_start', 'text'], sort=True)

        num_annotations = np.zeros(data.shape[0]) # count annos per token
        for col in annotator_cols:
            num_annotations += np.invert(data[col].isna())

        for doc in np.unique(data['doc_id']):
            # get tokens from this doc
            drows = data['doc_id'] == doc

            # get the annotation counts for this doc
            counts = num_annotations[drows]

            # check that all tokens have same number of annos
            if len(np.unique(counts)) > 1:
                print('Test data: we have some misaligned labels.')
                print(counts)

            if np.any(counts.values == 0):
                print('Removing document %s with no annos.' % doc)

        # remove any lines with no annos
        annotated_idxs = num_annotations >= 1
        data = data[annotated_idxs]

        # save the annos.csv
        data.to_csv(savepath + '/task1_test_annos.csv', columns=annotator_cols, index=False,
                    float_format='%.f', na_rep=-1)

        # save the text in same order
        data.to_csv(savepath + '/task1_test_text.csv', columns=['text'], header=False, index=False)

        # save the doc starts
        data.to_csv(savepath + '/task1_test_doc_start.csv', columns=['doc_start'], header=False, index=False)

        # save the annos.csv
        data.to_csv(savepath + '/task1_test_gt.csv', columns=['gold'], header=False, index=False)

        # 5. Create a file containing only the words for the task 2 validation set, i.e. like annos.csv with no annos.
        # Create ground truth CSV for task1_val_path, task1_test_path and task2_val_path but blank out the task_1 labels
        # (for tuning the LSTM for task 2)

        import csv
        eng_val = pd.read_csv(task2_val_path, delimiter=' ', usecols=[0,3], names=['text', 'gold'],
                              skip_blank_lines=True, quoting=csv.QUOTE_NONE)

        doc_starts = np.zeros(eng_val.shape[0])
        docstart_token = eng_val['text'][0]

        doc_starts[1:] = (eng_val['text'] == docstart_token)[:-1]
        eng_val['doc_start'] = doc_starts

        eng_val['tok_idx'] = eng_val.index
        eng_val = eng_val[eng_val['text'] != docstart_token] # remove all the docstart labels

        eng_val['gold'] = _map_ner_str_to_labels(eng_val['gold'])
        eng_val['gold'] = IOB_to_IOB2(eng_val['gold'].values)

        eng_val.to_csv(savepath + '/task2_val_gt.csv', columns=['gold'], header=False, index=False)
        eng_val.to_csv(savepath + '/task2_val_text.csv', columns=['text'], header=False, index=False)
        eng_val.to_csv(savepath + '/task2_val_doc_start.csv', columns=['doc_start'], header=False, index=False)


        # 6. Create a file containing only the words for the task 2 test set, i.e. like annos.csv with no annos.
        # Create ground truth CSV for task1_val_path, task1_test_path and task2_test_path but blank out the task_1 labels/
        eng_test = pd.read_csv(task2_test_path, delimiter=' ', usecols=[0,3], names=['text', 'gold'],
                               skip_blank_lines=True, quoting=csv.QUOTE_NONE)

        doc_starts = np.zeros(eng_test.shape[0])
        docstart_token = eng_test['text'][0]

        doc_starts[1:] = (eng_test['text'] == docstart_token)[:-1]
        eng_test['doc_start'] = doc_starts

        eng_test['tok_idx'] = eng_test.index
        eng_test = eng_test[eng_test['text'] != docstart_token] # remove all the docstart labels

        eng_test['gold'] = _map_ner_str_to_labels(eng_test['gold'])
        eng_test['gold'] = IOB_to_IOB2(eng_test['gold'].values)

        eng_test.to_csv(savepath + '/task2_test_gt.csv', columns=['gold'], header=False, index=False)
        eng_test.to_csv(savepath + '/task2_test_text.csv', columns=['text'], header=False, index=False)
        eng_test.to_csv(savepath + '/task2_test_doc_start.csv', columns=['doc_start'], header=False, index=False)


    # 7. Reload the data for the current run...
    print('loading annos for task1 test...')
    annos = pd.read_csv(savepath + '/task1_test_annos.csv', skip_blank_lines=False)

    print('loading text data for task1 test...')
    text = pd.read_csv(savepath + '/task1_test_text.csv', skip_blank_lines=False, header=None)

    print('loading doc_starts for task1 test...')
    doc_start = pd.read_csv(savepath + '/task1_test_doc_start.csv', skip_blank_lines=False, header=None)

    print('loading ground truth for task1 test...')
    gt_t = pd.read_csv(savepath + '/task1_test_gt.csv', skip_blank_lines=False, header=None)

    print('Unique labels: ')
    print(np.unique(gt_t))
    print(gt_t.shape)

    print('loading annos for task1 val...')
    annos_v = pd.read_csv(savepath + '/task1_val_annos.csv', skip_blank_lines=False)

    # remove any lines with no annos
    # annotated_idxs = np.argwhere(np.any(annos_v != -1, axis=1)).flatten()
    # annos_v = annos_v.iloc[annotated_idxs, :]

    annos = pd.concat((annos, annos_v), axis=0)
    annos = annos.fillna(-1)
    annos = annos.values
    print('loaded annos for %i tokens' % annos.shape[0])

    print('loading text data for task1 val...')
    text_v = pd.read_csv(savepath + '/task1_val_text.csv', skip_blank_lines=False, header=None)
    # text_v = text_v.iloc[annotated_idxs]

    text = pd.concat((text, text_v), axis=0)
    text = text.fillna(' ').values

    print('loading doc_starts for task1 val...')
    doc_start_v = pd.read_csv(savepath + '/task1_val_doc_start.csv', skip_blank_lines=False, header=None)
    # doc_start_v = doc_start_v.iloc[annotated_idxs]

    doc_start = pd.concat((doc_start, doc_start_v), axis=0).values

    print('Loading a gold valdation set for task 1')
    gt_blanks = pd.DataFrame(np.zeros(gt_t.shape[0]) - 1)
    gt_val_task1_orig = pd.read_csv(savepath + '/task1_val_gt.csv', skip_blank_lines=False, header=None)
    gt_val_task1 = pd.concat((gt_blanks, gt_val_task1_orig), axis=0).values

    print('not concatenating ground truth for task1 val')
    gt_v = pd.DataFrame(np.zeros(annos_v.shape[0]) - 1) # gt_val_task1_orig#
    #gt = pd.DataFrame(np.zeros(gt.shape[0]) - 1)  # gt_val_task1_orig#
    gt_v_real = pd.read_csv(savepath + '/task1_val_gt.csv', skip_blank_lines=False, header=None)
    #gt_v = gt_v.iloc[annotated_idxs]

    gt = pd.concat((gt_t, gt_v), axis=0).values
    gt_all = pd.concat((gt_t, gt_v_real), axis=0).values
    print('loaded ground truth for %i tokens' % gt.shape[0])

    print('loading text data for task 2 test')
    text_task2 = pd.read_csv(savepath + '/task2_test_text.csv', skip_blank_lines=False, header=None)
    text_task2 = text_task2.fillna(' ').values

    print('loading doc_starts for task 2 test')
    doc_start_task2 = pd.read_csv(savepath + '/task2_test_doc_start.csv', skip_blank_lines=False, header=None).values

    print('loading ground truth for task 2 test')
    gt_task2 = pd.read_csv(savepath + '/task2_test_gt.csv', skip_blank_lines=False, header=None).values

    # fix some invalid transitions in the data. These seem to be caused by some bad splitting between sentences.
    # restricted = [0, 3, 5, 7]
    # unrestricted = [2, 4, 6, 8]
    # outside = 1
    # for typeid, r in enumerate(restricted):
    #     for tok in range(len(annos)):
    #         # correct O->I transitions to O->B. There are some errors in the data itself.
    #         if tok == 0:
    #             #annos[tok, annos[tok] == r] = unrestricted[typeid]
    #             if gt[tok] == r:
    #                 gt[tok] = unrestricted[typeid]
    #             if gt_val_task1[tok] == r:
    #                 gt_val_task1[tok] = unrestricted[typeid]
    #         else:
    #             #if doc_start[tok]:
    #             #    annos[tok, (annos[tok] == r) & doc_start[tok]] = unrestricted[typeid]
    #             #else:
    #             #    annos[tok, (annos[tok] == r) & (annos[tok - 1] == outside)] = unrestricted[typeid]
    #
    #             if gt[tok] == r and (gt[tok - 1] == outside or doc_start[tok]):
    #                 gt[tok] = unrestricted[typeid]
    #             if gt_val_task1[tok] == r and (gt_val_task1[tok - 1] == outside or doc_start[tok]):
    #                 gt_val_task1[tok] = unrestricted[typeid]

    return gt, annos, doc_start, text, gt_task2, doc_start_task2, text_task2, \
           gt_val_task1, gt_all


def cap_number_of_workers(crowd, doc_start, max_workers_per_doc):
    print('Reducing number of workers per document to %i' % max_workers_per_doc)
    doc_start_idxs = np.where(doc_start)[0]
    for d, doc in enumerate(doc_start_idxs):
        valid_workers = crowd[doc] != -1
        worker_count = np.sum(valid_workers)
        if worker_count > max_workers_per_doc:
            valid_workers = np.argwhere(valid_workers).flatten()
            drop_workers = valid_workers[max_workers_per_doc:]
            # print('dropping workers %s' % str(drop_workers))
            if d+1 < len(doc_start_idxs):
                next_doc = doc_start_idxs[d + 1]
                crowd[doc:next_doc, drop_workers] = -1
            else:
                crowd[doc:, drop_workers] = -1

    used_workers = np.any(crowd != -1, 0)
    crowd = crowd[:, used_workers]

    return crowd

def split_dev_set(gt, crowd, text, doc_start):
    all_doc_starts = np.where(doc_start)[0]
    doc_starts = np.where(doc_start & (gt != -1))[0]
    dev_starts = doc_starts[100:]

    crowd_dev = []
    gt_dev = []
    text_dev = []
    doc_start_dev = []
    for dev_start in dev_starts:
        next_doc_start = np.where(all_doc_starts == dev_start)[0][0] + 1
        if next_doc_start < all_doc_starts.shape[0]:
            doc_end = all_doc_starts[next_doc_start]
        else:
            doc_end = all_doc_starts.shape[0]

        crowd_dev.append(crowd[dev_start:doc_end])

        gt_dev.append(np.copy(gt[dev_start:doc_end]))
        gt[dev_start:doc_end] = -1

        text_dev.append(text[dev_start:doc_end])
        doc_start_dev.append(doc_start[dev_start:doc_end])

    # # include a sample of the unlabelled data
    # s = 7900 # total number of documents in dev set
    # moreidxs = np.argwhere(gt == -1)[:, 0]
    # deficit = s - 60 #number of labelled dev docs
    # moreidxs = moreidxs[:np.argwhere(np.cumsum(doc_start[moreidxs])==deficit)[0][0]]
    #
    # crowd_dev.append(crowd[moreidxs])
    # gt_dev.append(gt[moreidxs])
    # text_dev.append(text[moreidxs])
    # doc_start_dev.append(doc_start[moreidxs])

    crowd_dev = np.concatenate(crowd_dev, axis=0)
    gt_dev = np.concatenate(gt_dev, axis=0)
    text_dev = np.concatenate(text_dev, axis=0)
    doc_start_dev = np.concatenate(doc_start_dev, axis=0)

    return gt, crowd_dev, gt_dev, doc_start_dev, text_dev

def load_arg_sentences(debug_size=0, regen_data=False, second_batch_workers_only=False, gold_labelled_only=False,
                       max_workers_per_doc=5):

    data_dir = os.path.join(data_root_dir, 'argmin_LMU')
    if not regen_data and os.path.exists(os.path.join(data_dir, 'evaluation_gold.csv')):
        #reload the data for the experiments from cache files
        gt = pd.read_csv(os.path.join(data_dir, 'evaluation_gold.csv'), usecols=[1]).values.astype(int)
        crowd = pd.read_csv(os.path.join(data_dir, 'evaluation_crowd.csv')).values[:, 1:].astype(int)
        doc_start = pd.read_csv(os.path.join(data_dir, 'evaluation_doc_start.csv'), usecols=[1]).values.astype(int)
        text = pd.read_csv(os.path.join(data_dir, 'evaluation_text.csv'), usecols=[1]).values

        if second_batch_workers_only:
            crowd = crowd[:, 26:]

        if gold_labelled_only:
            idxs = gt.flatten() != -1
            gt = gt[idxs]
            crowd = crowd[idxs, :]
            doc_start = doc_start[idxs]
            text = text[idxs]

        if max_workers_per_doc > 0:
            crowd = cap_number_of_workers(crowd, doc_start, max_workers_per_doc)


        # split dev set
        gt, crowd_dev, gt_dev, doc_start_dev, text_dev = split_dev_set(gt, crowd, text, doc_start)

        if debug_size:
            gt = gt[:debug_size]
            crowd = crowd[:debug_size]
            doc_start = doc_start[:debug_size]
            text = text[:debug_size]

        return gt, crowd, doc_start, text, crowd_dev, gt_dev, doc_start_dev, text_dev

    # else generate the correct file format from the original dataset files
    expert_file = os.path.join(data_dir, 'expert_corrected_disagreements.csv')

    gold_text = pd.read_csv(expert_file, sep=',', usecols=[6]).values
    gold_doc_start = pd.read_csv(expert_file, sep=',', usecols=[1]).values
    gold = pd.read_csv(expert_file, sep=',', usecols=[7]).values.astype(int).flatten()

    crowd_on_gold_file = os.path.join(data_dir, 'crowd_on_expert_labelled_sentences.csv')
    crowd_on_gold = pd.read_csv(crowd_on_gold_file, sep=',', usecols=range(2,28)).values

    crowd_no_gold_file = os.path.join(data_dir, 'crowd_on_sentences_with_no_experts.csv')
    crowd_without_gold = pd.read_csv(crowd_no_gold_file, sep=',', usecols=range(2,100)).values
    nogold_doc_start = pd.read_csv(crowd_no_gold_file, sep=',', usecols=[1]).values
    nogold_text = pd.read_csv(crowd_no_gold_file, sep=',', usecols=[100]).values

    print('Number of tokens = %i' % crowd_without_gold.shape[0])

    # some of these data points may have no annos
    valididxs = np.any(crowd_without_gold != -1, axis=1)
    crowd_without_gold = crowd_without_gold[valididxs, :]
    doc_start = nogold_doc_start[valididxs]
    text = nogold_text[valididxs]

    print('Number of crowd-labelled tokens = %i' % crowd_without_gold.shape[0])

    N = crowd_without_gold.shape[0]

    # now line up the gold sentences with the complete set of crowd data
    crowd = np.zeros((N, crowd_on_gold.shape[1] + crowd_without_gold.shape[1]), dtype=int) - 1
    crowd[:, crowd_on_gold.shape[1]:] = crowd_without_gold

    # crowd_labels_present = np.any(crowd != -1, axis=1)
    # N_withcrowd = np.sum(crowd_labels_present)

    gt = np.zeros(N) - 1

    gold_docs = np.split(gold_text, np.where(gold_doc_start == 1)[0][1:], axis=0)
    gold_gold = np.split(gold, np.where(gold_doc_start == 1)[0][1:], axis=0)
    gold_crowd = np.split(crowd_on_gold, np.where(gold_doc_start == 1)[0][1:], axis=0)

    nogold_docs = np.split(text, np.where(nogold_doc_start == 1)[0][1:], axis=0)
    for d, doc in enumerate(gold_docs):

        print('matching gold doc %i of %i' % (d, len(gold_docs)))

        loc_in_nogold = 0

        for doc_nogold in nogold_docs:
            if np.all(doc == doc_nogold):
                len_doc_nogold = len(doc_nogold)
                break
            else:
                loc_in_nogold += len(doc_nogold)

        locs_in_nogold = np.arange(loc_in_nogold, len_doc_nogold+loc_in_nogold)
        gt[locs_in_nogold] = gold_gold[d]
        crowd[locs_in_nogold, :crowd_on_gold.shape[1]] = gold_crowd[d]

    # we need to flip 3 and 4 to fit our scheme here
    ICon_idxs = gt == 4
    BCon_idxs = gt == 3
    gt[ICon_idxs] = 3
    gt[BCon_idxs] = 4

    ICon_idxs = crowd == 4
    BCon_idxs = crowd == 3
    crowd[ICon_idxs] = 3
    crowd[BCon_idxs] = 4


    # save files for our experiments with the tag 'evaluation_'
    pd.DataFrame(gt).to_csv(os.path.join(data_dir, 'evaluation_gold.csv'))
    pd.DataFrame(crowd).to_csv(os.path.join(data_dir, 'evaluation_crowd.csv'))
    pd.DataFrame(doc_start).to_csv(os.path.join(data_dir, 'evaluation_doc_start.csv'))
    pd.DataFrame(text).to_csv(os.path.join(data_dir, 'evaluation_text.csv'))

    if second_batch_workers_only:
        crowd = crowd[:, 26:]

    if gold_labelled_only:
        idxs = gt.flatten() != -1
        gt = gt[idxs]
        crowd = crowd[idxs, :]
        doc_start = doc_start[idxs]
        text = text[idxs]

    gt = gt.astype(int)

    if max_workers_per_doc > 0:
        crowd = cap_number_of_workers(crowd, doc_start, max_workers_per_doc)

    gt, crowd_dev, gt_dev, doc_start_dev, text_dev = split_dev_set(gt, crowd, text, doc_start)

    if debug_size:
        gt = gt[:debug_size]
        crowd = crowd[:debug_size]
        doc_start = doc_start[:debug_size]
        text = text[:debug_size]

    return gt, crowd, doc_start, text, crowd_dev, gt_dev, doc_start_dev, text_dev
