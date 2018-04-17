from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import util

import numpy as np

import hmm
import re


#train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
#test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))


upper = "[A-Z]"
lower = "a-z"
punc = "[,.;:?!()]"
quote = "[\'\"]"
digit = "[0-9]"
multidot = "[.][.]"
hyphen = '[/]'
dollar = '[$]'
at = '[@]'

all_cap = upper + "+$"
all_digit = digit + "+$"

contain_cap = ".*" + upper + ".*"
contain_punc = ".*" + punc + ".*"
contain_quote = ".*" + quote + ".*"
contain_digit = ".*" + digit + ".*"
contain_multidot = ".*" + multidot + ".*"
contain_hyphen = ".*" + hyphen + ".*"
contain_dollar = ".*" + dollar + ".*"
contain_at = ".*" + at + ".*"

cap_period = upper + "\.$"


list_reg = [contain_cap, contain_punc, contain_quote,
            contain_digit, cap_period, punc, quote, digit,
            contain_multidot, contain_hyphen, contain_dollar, contain_at]

def reg_features(word, start = ''):
    res = []
    for p in list_reg:
        if re.compile(p).match(word): 
            res.append(start + p + '=TRUE')
        else:
            res.append(start + p + '=FALS')

    return res


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]

    features.extend(reg_features(word))
    
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
        features.extend(reg_features(word1, '-1'))
    else:
        features.append('BOS')

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
        features.extend(reg_features(word1, '+1'))
    else:
        features.append('EOS')

    # window size
    L = 3
    for l in range(2, L+1, 1):
        if i-l >= 0:
            word1 = sent[i-l][0]
            postag1 = sent[i-l][1]
            features.extend([
                '-%d:word.lower=' % l + word1.lower(),
                '-%d:word.istitle=%s' % (l, word1.istitle()),
                '-%d:word.isupper=%s' % (l, word1.isupper()),
                '-%d:postag=' % l + postag1,
                '-%d:postag[:2]=' % l + postag1[:2],
            ])
            features.extend(reg_features(word1, '-%d' % l))

        if i + l < len(sent):
            word1 = sent[i+l][0]
            postag1 = sent[i+l][1]
            features.extend([
                '+%d:word.lower='  % l + word1.lower(),
                '+%d:word.istitle=%s' % (l, word1.istitle()),
                '+%d:word.isupper=%s' % (l, word1.isupper()),
                '+%d:postag=' % l + postag1,
                '+%d:postag[:2]=' % l + postag1[:2],
            ])
            features.extend(reg_features(word1, '+%d' % l))

    # bigram and trigram
    if i > 0:
        features.append('bigram-1=' + sent[i-1][0].lower() + '_' + word.lower())
        if i > 1:
            features.append('trigram-2=' + sent[i-2][0].lower() + '_' + sent[i-1][0].lower() + '_' + word.lower() )

    if i < len(sent) - 1:
        features.append('bigram+1=' + sent[i+1][0].lower() + '_' + word.lower())
        if i < len(sent) - 2:
            features.append('trigram+2=' + sent[i+2][0].lower() + '_' + sent[i+1][0].lower() + '_' + word.lower() )

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


#X_train = [sent2features(s) for s in train_sents]
#y_train = [sent2labels(s) for s in train_sents]

#X_test = [sent2features(s) for s in test_sents]
#y_test = [sent2labels(s) for s in test_sents]



#for xseq, yseq in zip(X_train, y_train):
#    trainer.append(xseq, yseq)


def get_data(sentences, features, labels, list_labels = None, flag = None):
    """
    :params list_labels: list of labels. If = None then use labels in sentences
    :params flag: flag[i] = True -> not use sentence i
    """


    #trainer.set_params({
        #'c1': 1.0,   # coefficient for L1 penalty
        #'c2': 1e-3,  # coefficient for L2 penalty
        #'max_iterations': 1000,  # stop earlier

        # include transitions that are possible, but not observed
        #'feature.possible_transitions': True
    #})

    a = []

    inv_f = {v: k for k, v in features.items()}

    for i, sen in enumerate(sentences):
        if flag != None and flag[i]: continue # skip sentece with flag
        x = []
        y = []
        for ins in sen:
            #print ins.features, ins.labels
            w = ins.features[0]
            if w in inv_f:
                x.append(inv_f[w])
            else:
                x.append("OOVWORD")

            y.append(str(ins.label))
        x = nltk.pos_tag(x)
        x = [word2features(x, j) for j in range(len(x))]
        #x = map(str, util.get_words(sen, features))
        #y = map(str, util.get_lab(sen))
        #trainer.append(x, y)
        if list_labels == None:
            a.append((x,y))
        else:
            a.append((x, map(str, map(int, list_labels[i]) ) ))
    return a

def train(a):
    """
    :param a: data from get_data
    """
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.select('lbfgs')
    trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1.0,  # coefficient for L2 penalty
    'max_iterations': 100000000,  # stop earlier
    'feature.possible_transitions': True,
    'feature.possible_states': True,
    'max_linesearch': 20,
    'num_memories': 6,
    'period':10
    })

    for (x,y) in a:
        trainer.append(x, y)
    trainer.train('test.crfsuite')

    tagger = pycrfsuite.Tagger()
    tagger.open('test.crfsuite')
    return tagger



def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

#y_pred = [tagger.tag(xseq) for xseq in X_test]

#print(bio_classification_report(y_test, y_pred))


def get_res(t, a):
    res = [t.tag(a[i][0]) for i in range(len(a))]
    return res


def list_s2i(a):
    res = []
    for x in a:
        res.append(map(int,x))
    return res


def eval_tagger_train(t, a, labels):
    res = get_res(t, a)
    b, c = zip(*a)
    hmm.eval_seq_train(list_s2i(c), list_s2i(res), labels)


def sample_labels(sen_pos, trans_pos, rs):
    n = len(sen_pos)
    
    first_lab = np.nonzero(rs.multinomial(1, sen_pos[0]))[0][0]
    res = [first_lab]

    for i in range(1, n, 1):
        prev = res[i-1]
        join_dis = trans_pos[i-1]
        next = np.nonzero(rs.multinomial(1, join_dis[prev]))[0][0]
        res.append(next)

    return res


def sample_data(hc, c = 5, seed = 0):
    """
    :param c: number of copies
    sample labels from posterior
    """
    rs = np.random.RandomState(seed)
    res = []
    lab = []
    for i, sen in enumerate(hc.data.sentences):
        sen_pos = hc.sen_posterior[i]
        if len(sen_pos) == 0: continue
        trans_pos = hc.trans_posterior[i]
        for cnt in range(c):
            res.append(sen)
            #sample a label for each token:
            labels = sample_labels(sen_pos, trans_pos, rs)
            lab.append(labels)

    return (res, lab)
