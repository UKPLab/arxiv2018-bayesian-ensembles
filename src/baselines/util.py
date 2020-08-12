'''


'''
import numpy
import numpy as np
import os
#import re
import matplotlib.pyplot as plt
import pickle
import csv
#import shutil

class instance:
    """
    an instance
    """

    def __init__(self, features, label, word = None):
        self.features = features
        self.label = label
        if word != None:
            self.word = word


upper = "[A-Z]"
lower = "a-z"
punc = "[,.;:?!()]"
quote = "[\'\"]"
digit = "[0-9]"

all_cap = upper + "+$"
all_digit = digit + "+$"

contain_cap = ".+" + upper + ".+"
contain_punc = ".+" + punc + ".+"
contain_quote = ".+" + quote + ".+"
contain_digit = ".+" + digit + ".+"

cap_period = upper + "\.$"


list_reg = [all_cap, all_digit, contain_cap, contain_punc, contain_quote,
            contain_digit, cap_period, upper, lower, punc, quote, digit]

list_prefix = ['anti', 'auto', 'de', 'dis', 'down', 'extra', 'hyper', 'il', 'im', 'in', 'ir', 'inter', 'mega', 'mid',
               'mis', 'non', 'over', 'out', 'post', 'pre', 'pro', 're', 'semi', 'sub', 'super', 'tele', 'trans',
               'ultra', 'un', 'under', 'up']

list_suffix = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist', 'ity', 'ty', 'ment', 'ness', 'ship',
               'sion', 'tion', 'able', 'ible', 'al', 'esque', 'ful', 'ic', 'ical', 'ious', 'ous', 'ish', 'ive'
               'less', 'ing', 'ed', 'land', 'burg']


def get_features(word, prev, next):
    word = word.strip()
    a = []
    if len(word) > 0:
        # for p in list_reg:
        #    if re.compile(p).match(word): a.append("*REG" + p)
        if word[0].isupper():
            a.append("*U")
        if word[0].isdigit():
            a.append("*D")

        #a.append("*PREV" + prev.lower())
        #a.append("*NEXT" + next.lower())

        # for p in list_prefix:
        #     if word.startswith(p):
        #         a.append(p)
        #
        # for p in list_suffix:
        #     if word.endswith(p):
        #         a.append(p)
        return a

    if len(word) > 0:
        #if word[0].isupper():
        #    a.append("*U")
        #if word[0].islower():
        #    a.append("*L")
        #if word[0].isdigit():
        #    a.append("*D")
        #a.append(word[0])
        #a.append(word[-1])
        if len(word) > 1:
            a.append(word[:2])
            a.append(word[-2:])
            if len(word) > 2:
                a.append(word[:3])
                a.append(word[-3:])

    return a


def process_word(word):
    """
    to standardize word to put in list of features
    :param word:
    :return:
    """
    # return word.lower()
    return word


def get_first_word(s):
    x = s.strip().split()
    if len(x) < 1:
        return ""
    return process_word(x[0])


def get_prev_next(input, i):
    n = len(input)
    if i == 0:
        return ("*ST", get_first_word(input[i + 1]))
    if i == n - 1:
        return (get_first_word(input[i - 1]), "*EN")
    return (get_first_word(input[i - 1]), get_first_word(input[i + 1]))


def build_index(input):

    list_labels = []
    list_features = []

    for i, line in enumerate(input):
        a = line.strip().split()
        #print a
        if a == []:
            continue
        # last word is label
        list_labels.append(a[-1])
        for f in a[:-1]:
            list_features.append(process_word(f))

        prev, next = get_prev_next(input, i)
        list_features.extend(get_features(a[0], prev, next))

    # get unique labels and features
    list_labels = sorted(list(set(list_labels)))
    list_features = sorted(list(set(list_features)))
    #list_labels = list(set(list_labels))
    #list_features = list(set(list_features))

    # maps from word to labels/features
    labels = {}
    features = {}
    for i, l in enumerate(list_labels):
        labels[l] = i + 1  # label 0 = unseen label
    for i, f in enumerate(list_features):
        features[f] = i + 1  # features 0 = OOV

    return features, labels


def extract(input, features, labels, keep_word=False):
    sentences = []
    sentence = []
    for i, line in enumerate(input):
        a = line.strip().split()
        if a == []:
            if sentence != []:
                sentences.append(sentence)
                sentence = []
            continue
        list_f = []

        prev, next = get_prev_next(input, i)

        if process_word(a[0]) in features:
            list_f.append(features[process_word(a[0])])

            for f in get_features(a[0], prev, next):
                if f in features:
                    list_f.append(features[f])
        else:
            for f in get_features(a[0], prev, next):
                if f in features:
                    list_f.append(features[f])

        if list_f == []:
            list_f = [0]

        label = labels[a[-1]] if a[-1] in labels else 0
        if keep_word:
            i = instance(list_f, label, a[0])
        else:
            i = instance(list_f, label)
        sentence.append(i)
    return sentences


def load(filename, per=0.8):
    file = open(filename)
    input = list(file)
    file.close()

    l = int(len(input) * per)
    features, labels = build_index(input[:l])
    train = extract(input[:l], features, labels)
    test = extract(input[l:], features, labels)

    return train, test, features, labels


def get_obs(sentence):
    """
    get the seq of observations, each observation = list of features
    :param sentence:
    :return:
    """
    res = []
    for i in sentence:
        res.append(i.features)
    return res


def get_ff(sentence):
    """
    get the seq of the first feature
    :param sentence:
    :return:
    """
    res = []
    for i in sentence:
        res.append(i.features[0])
    return res

def get_word_list(sentence, features):
    inv_f = {v: k for k, v in list(features.items())}
    res = []
    for i in sentence:
        if len(i.features) > 0:
            if i.features[0] in inv_f:
                res.append(inv_f[i.features[0]])
            else:
                res.append("*EMPTY*")
        else:
            res.append("*NOFEA*")
    return res


def get_word_list2(sentence, inv_f):
    res = []
    for i in sentence:
        if len(i.features) > 0:
            if i.features[0] in inv_f:
                res.append(inv_f[i.features[0]])
            else:
                res.append("*EMPTY*")
        else:
            res.append("*NOFEA*")
    return res

def get_lab(sentence):
    """
    get the seq of labels
    :param sentence:
    :return:
    """
    res = []
    for i in sentence:
        res.append(i.label)
    return res


def get_lab_name(sen, labels):
    """
    sen = list of numerical labels
    """
    inv_l = {v: k for k, v in list(labels.items())}
    res = ""
    for i in sen:
        if i in inv_l:
            res = res + " " + inv_l[i]
        else:
            res = res + " *NEWLABEL"
    return res
    
def get_lab_name_list(sen, labels):
    """
    sen = list of numerical labels
    """
    inv_l = {v: k for k, v in list(labels.items())}
    res = []
    for i in sen:
        if i in inv_l:
            res.append(inv_l[i])
        else:
            res.append("*NEWLABEL")
    return res


def get_lab_name_list2(sen, inv_l):
    """
    sen = list of numerical labels
    """
    res = []
    for i in sen:
        if i in inv_l:
            res.append(inv_l[i])
        else:
            res.append("*NEWLABEL")
    return res



def get_all_lab(sentences):
    res = []
    for sentence in sentences:
        res.append(get_lab(sentence))
    return res


def get_words(sen, features):
    inv_f = {v: k for k, v in list(features.items())}
    res = ""
    for i in sen:
        if len(i.features) > 0:
            if i.features[0] in inv_f:
                res = res + " " + (inv_f[i.features[0]])
            else:
                res = res + (" *EMPTY*")
        else:
            res = res + (" *NOFEA*")
    return res


def print_sentence_labels(sentence):
    for ins in sentence:
        print(ins.label, end=' ')
    print("")


class crowdlab:
    """
    a sentence labeled by crowd
    """

    def __init__(self, wid, sid, sen):
        """

        :param sen: list of labels
        :param wid: worker id
        :param sid: sentence id
        """
        self.sen = sen
        self.wid = wid
        self.sid = sid


class crowd_data:
    """
    a dataset with crowd labels
    """

    def __init__(self, sentences, crowdlabs):
        """

        :param sentences: list of sen
        :param crowdlabs: list of list of crowdlab
        each "list of crowdlab" is a list of crowd labels for a particular sentence
        """
        self.sentences = sentences
        self.crowdlabs = crowdlabs

    def get_labs(self, i, j):
        """
        return all labels for sentence i, position j
        """
        res = []
        for c in self.crowdlabs[i]:
            res.append(c.sen[j])
        return res

    def get_lw(self, i, j):
        """
        return all labels/wid for sentence i, position j
        """
        res = []
        for c in self.crowdlabs[i]:
            res.append( (c.sen[j], c.wid))
        return res


class simulator:

    def __init__(self, sentences, features, labels, seed, nwk=5):
        self.rs = np.random.RandomState(seed)
        self.sentences = sentences
        self.features = features
        self.labels = labels

        self.wa = [0.7, 0.8, 0.2, 0.9, 0.9]
        self.f = [0.3, 0.7, 0.01, 0.8, 0.2]

    def sim_label(self, true_lab, prob):
        """
        :param true_lab:
        :param prob:
        :return:
        """
        if self.rs.rand() < prob:
            return true_lab
        else:
            return self.rs.choice(list(self.labels.values()))

    def sim_labels(self, sentence, wid):
        sen = []
        for ins in sentence:
            sen.append(self.sim_label(ins.label, self.wa[wid]))
        return sen

    def sim_dec(self, wid):
        """
        simulate the decision to label
        :param wid:
        :return:
        """
        if self.rs.rand() < self.f[wid]:
            return True
        else:
            return False

    def simulate(self):
        """

        :param train:
        :param features:
        :param labels:
        :return:
        """
        crowd_sentences = []                                               # all the crowdlab

        for sid, sentence in enumerate(self.sentences):
            # list of crowdlab for this sentence
            cls = []
            for wid in range(5):
                if self.sim_dec(wid) or (cls == [] and wid == 4):
                    sen = self.sim_labels(sentence, wid)
                    cl = crowdlab(wid, sid, sen)
                    cls.append(cl)
            crowd_sentences.append(cls)

        self.cd = crowd_data(self.sentences, crowd_sentences)


def read_rod(dirname="ground_truth"):
    """
    read data in the form provided by Rodrigues et. al. 2014
    all files in dir
    :return:
    """

    # first pass: read all and build index
    input = []
    for file in sorted(os.listdir(dirname)):
        if file.endswith(".txt"):
            f = open(os.path.join(dirname, file))
            l = list(f)
            input.extend(l)
            f.close()

    features, labels = build_index(input)

    # second pass:
    sens = []                   # all sentences in all docs
    docs = {}                   # position of each doc in sens
    for file in sorted(os.listdir(dirname)):
        if file.endswith(".txt"):
            f = open(os.path.join(dirname, file))
            l = extract(list(f), features, labels)
            docs[file] = len(sens)
            sens.extend(l)
            f.close()

    return sens, features, labels, docs


def cmp_list(l1, l2):
    if len(l1) != len(l2):
        return False
    for i, j in zip(l1, l2):
        if i != j:
            return False
    return True


def align_sen(a, b):
    """
    len(a) must < len(b)
    :param a:
    :param b:
    :return:
    """
    res = []
    for i in range(len(a)):
        if a[i].features[0] == b[i].features[0]:
            res.append(b[i].label)
        else:
            found = False
            for j in range(i, len(b), 1):
                if a[i].features[0] == b[i].features[0]:
                    res.append(b[i].label)
                    found = True
                    break
            if not found:
                res.append(0)
    return res


def read_workers_rod(all, features, labels, docs, dirname='mturk_train_data'):
    """
    read workers labels
    :param all:         all sentences
    :param features:
    :param labels:
    :param docs:
    :param dirname:
    :return:
    """
    cls = []
    for i in range(len(all)):
        cls.append([])

    dic_workers = {}                                # worker name -> worker id

    cnt = 0
    c2 = 0
    id_err = 0

    for dir in os.listdir(dirname):                 # dir = a worker
        if not dir.startswith("."):
            wid = len(dic_workers)
            dic_workers[dir] = wid
            fullpath = os.path.join(dirname, dir)
            for file in os.listdir(fullpath):       # file = a doc
                if file.endswith(".txt"):
                    f = open(os.path.join(fullpath, file))
                    l = extract(list(f), features, labels)
                    # position to stitch crowd labels to
                    pos = docs[file]

                    for sen in l:
                        try:
                            if not cmp_list(get_ff(all[pos]), get_ff(sen)):
                                # print "Not equal", fullpath, file
                                # if len(all[pos]) != len(sen):
                                # if len(all[pos]) +1 != len(sen):
                                    # print len(all[pos]), len(sen)
                                cnt += 1
                                #labs = align_sen(all[pos], sen)
                                #cls[pos].append(crowdlab(wid, pos, labs))
                            # print get_words(all[pos], features)
                            # print get_words(sen, features)
                            # print get_ff(all[pos]), get_words(all[pos], features)
                            # print get_ff(sen), get_words(sen, features)
                            #cnt += 1
                            else:
                                cls[pos].append(
                                    crowdlab(wid, pos, get_lab(sen)))
                                c2 += 1
                        except IndexError:
                            # print "IndexError", fullpath, file, pos
                            #cnt += 1
                            id_err += 1

                        pos += 1

    print(cnt, id_err, c2)
    cd = crowd_data(all, cls)
    print("num features = ", len(features))
    for i in range(len(all)):
        sen = all[i]
        cl = cls[i]
        if len(cl) == 0:
            all[i] = []

    cd = crowd_data(all, cls)
    return cd


def process_test(filename):
    f = open(filename)

    split_string = "-DOCSTART- -X- O O"
    docs = []
    current_doc = []
    for line in f:
        if line.startswith(split_string):
            if len(current_doc) > 0:
                docs.append(current_doc)
            current_doc = []
        else:
            current_doc.append(line)

    if len(current_doc) > 0:
        docs.append(current_doc)
    i = 0
    for doc in docs:
        i += 1
        name = str(i) + ".txt"
        g = open("test/" + name, 'w')
        for line in doc:
            g.write(line)
        g.close()

    f.close()


def process_test2():

    for i in range(1, 217, 1):
        f = list(open('test/' + str(i) + ".txt"))
        g = list(open('test_censored/' + str(i) + ".txt"))
        h = open('test1/' + str(i) + ".txt", 'w')

        for x, y in zip(f, g):
            if len(x.strip()) == 0:
                h.write(x)
            else:
                word = x.split()[0]
                lab = y.strip()
                h.write(word + ' ' + lab + "\n")
        h.close()


def mv(l):
    if len(l) == 0:
        return 0
    m = max(l)
    a = [0] * (m + 1)
    for i in l:
        a[i] += 1
    return np.argmax(a)


def mv_cd(cd, labels, n_workers=100, smooth=0.001, print_star = False):
    """

    :param cd: crowd_data
    :return:
    """
    #X = 0
    #Y = 0
    #Z = 0
    sentences = []
    w_correct = smooth + np.zeros((n_workers, ))
    w_wrong = smooth + np.zeros((n_workers,))

    for sen, clab in zip(cd.sentences, cd.crowdlabs):
        new_sen = []
        count = []
        for i in range(len(sen)):
            count.append([])
        if len(clab) == 0:
            if print_star:
                print("*", end=' ')
            sentences.append([])
            continue
        for c in clab:  # c = labels for the sentence of a worker
            # i = position in sentence, l = label given by worker c
            for i, l in enumerate(c.sen):
                count[i].append(l)

        for i, inst in enumerate(sen):
            mv_lab = mv(count[i])
            new_inst = instance(inst.features, mv_lab)
            new_sen.append(new_inst)

            # update worker correctness counts
            for c in clab:
                if mv_lab == c.sen[i]:
                    w_correct[c.wid] += 1
                else:
                    w_wrong[c.wid] += 1

        # print count
        sentences.append(new_sen)
        #x, y, z = hmm.eval_ner(get_lab(sen), get_lab(new_sen), labels)
        #X += x
        #Y += y
        #Z += z

        # return sentences

    # print X, Y, Z
    return sentences


def cal_workers_true_acc(cd, true_labs, ne=3, n_workers=47, print_out=False, return_ss=False, smooth_0=10.0, smooth_1=10, w = 0, return_list = False):
    """

    :param cd:
    :param ne: id of the no entity label
    :return: accuracy of each worker conditioned on the class
    
    w and return_list: measure a particular worker
    """

    # correct_0 = np.zeros((47,))     # correct when true class is 0
    correct_1 = np.zeros((10, 47))  # when true class is 1
    #wrong_0 = np.zeros((47,))
    wrong_1 = np.zeros((10, 47))

    list_true = []
    list_l = []
    for sen, clab, true_lab in zip(cd.sentences, cd.crowdlabs, true_labs):
        #true_lab = get_lab(sen)
        for c in clab:
            wid = c.wid
            for i, l in enumerate(c.sen):
                # if true_lab[i] == ne:
                #    if true_lab[i] == l:
                #        correct_0[wid] += 1
                #    else:
                #        wrong_0[wid] += 1
                # else:
                if true_lab[i] == l:
                    correct_1[true_lab[i]][wid] += 1
                else:
                    wrong_1[true_lab[i]][wid] += 1
                if wid == w:
                    list_true.append(true_lab[i])
                    list_l.append(l)

    if return_list: return (list_true, list_l)
    # if print_out:
        # for i in range(47):
            # print i, 0, correct_0[i], wrong_0[i], correct_0[i] * 1.0 / (correct_0[i] + wrong_0[i])
            # print i, 1, correct_1[i], wrong_1[i], correct_1[i] * 1.0 /
            # (correct_1[i] + wrong_1[i])

    sen = np.zeros((10, 47))
    #spe = np.zeros((47,))

    for w in range(47):
        #spe[i] =  (correct_0[i] * 1.0 + smooth) / (correct_0[i] + wrong_0[i] + smooth*2)
        #sen[:,i] = (correct_1[:,i] * 1.0 + smooth) / (correct_1[:,i] + wrong_1[:,i] + smooth*2)
        # count number of correct ans given the class is positive (not
        # non-entity)
        sum_pos_correct = 0
        sum_pos_wrong = 0
        for i in range(10):
            if i != ne:
                sum_pos_correct += correct_1[i, w]
                sum_pos_wrong += wrong_1[i, w]

        for i in range(10):
            if i == ne:
                sen[i, w] = (correct_1[i, w] * 1.0 + smooth_1) / \
                    (correct_1[i, w] + wrong_1[i, w] + smooth_0 + smooth_1)
            else:
                sen[i, w] = (sum_pos_correct * 1.0 + smooth_1) / \
                    (sum_pos_correct + sum_pos_wrong + smooth_0 + smooth_1)

    if return_ss:
        return sen
    else:
        # return (correct_0, correct_1, wrong_0, wrong_1)
        return correct_1, wrong_1

def plot_e(list_d, ymax = 0.1):
    for d in list_d:
        plt.figure()
        plt.ylim(0, ymax)
        plt.plot(d)



def compare_t(t1, t2, labels, thresh = 0.1):
    inv_l = {v:k for (k,v) in list(labels.items())}
    n, m = t1.shape
    res = []
    for i in range(n):
        for j in range(m):
            res.append(abs(t1[i][j] - t2[i][j]))
            if abs(t1[i][j] - t2[i][j]) > thresh:
                if i > 0 and j > 0:
                    print(inv_l[i], inv_l[j], end=' ')
                print(i, j, t1[i][j], t2[i][j], (t1[i][j] > t2[i][j]))

    return res


# output words and labels
def output_sens(out_file, sens, features, labels):
    """
    output the word and label in CoNLL format to run Lample
    """
    inv_f = {v:k for (k,v) in list(features.items())}
    inv_l = {v:k for (k,v) in list(labels.items())}

    f = open(out_file, 'w')
    for sen in sens:
        for ins in sen:
            f.write(inv_f[ins.features[0]] + " " +  inv_l[ins.label] + "\n")
        f.write("\n")
    f.close()
    


def output_sens2(out_file, sens, res, features, labels):
    """
    output the word and label in CoNLL format to run Lample
    use res for labels
    """
    inv_f = {v:k for (k,v) in list(features.items())}
    inv_l = {v:k for (k,v) in list(labels.items())}

    f = open(out_file, 'w')
    f.write('@ O\n\n')
    for sen, lab in zip(sens, res):
        for ins, l in zip(sen, lab):
            f.write(inv_f[ins.features[0]] + " " +  inv_l[int(l)] + "\n")
        f.write("\n")
    f.close()


def output_sens_pico(out_file, sens, res, features, labels):
    """
    output the word and label in CoNLL format to run Lample
    use res for labels
    """
    inv_f = {v:k for (k,v) in list(features.items())}
    inv_l = {v:k for (k,v) in list(labels.items())}

    f = open(out_file, 'w')
    #f.write('@ O\n\n')
    for sen, lab in zip(sens, res):
        if sen == []: continue
        for ins, l in zip(sen, lab):
            #f.write(inv_f[ins.features[0]] + " " +  inv_l[int(l)] + "\n")
            f.write(inv_f[ins.features[0]] + " " + str(int(l)) + "\n")
        f.write("\n")
    f.close()

def to_lample(cd, features, labels):
    """
    convert to Lample format
    to run directly
    """
    inv_f = {v:k for (k,v) in list(features.items())}
    inv_l = {v:k for (k,v) in list(labels.items())}

    res = [[['@', 'O']]]
    wids = [0]
    for sen, clabs in zip(cd.sentences, cd.crowdlabs):
        #text_sen = get_word_list(sen, features) VERY SLOW
        text_sen = []
        for ins in sen:
            text_sen.append(inv_f[ins.features[0]])
        if text_sen == []: continue
        for cl in clabs:
            crowd_lab = []
            for ins in cl.sen:
                crowd_lab.append(inv_l[ins])
            res.append( list(map(list, list(zip(text_sen, crowd_lab)) )) )
            wids.append(cl.wid)
            #if len(res) > 100:
            #    return (res, wids)
            #if len(res) % 1000 == 0: print len(res)

    return (res, wids)

def to_lample_pico(cd, features, labels):
    """
    convert to Lample format
    to run directly
    """
    inv_f = {v:k for (k,v) in list(features.items())}
    inv_l = {v:k for (k,v) in list(labels.items())}

    res = []
    wids = []
    for sen, clabs in zip(cd.sentences, cd.crowdlabs):
        #text_sen = get_word_list(sen, features) VERY SLOW
        text_sen = []
        for ins in sen:
            text_sen.append(str(inv_f[ins.features[0]], 'utf-8'))
        if text_sen == []: continue
        for cl in clabs:
            crowd_lab = []
            for ins in cl.sen:
                crowd_lab.append(str(ins))
            res.append( list(map(list, list(zip(text_sen, crowd_lab)) )) )
            wids.append(cl.wid)
            #if len(res) > 100:
            #    return (res, wids)
            #if len(res) % 1000 == 0: print len(res)

    return (res, wids)


def make_val_test_data(use_set = 'val'):
    d = pickle.load( open('val_test.pkl') )
    a_val = d['a_val']
    a_test = d['a_test']
    if use_set == 'val':
        a = a_val
    else:
        a = a_test 


    for f in os.listdir('task1/' + use_set + '/ground_truth'):
        if f not in a:
            os.remove('task1/' + use_set + '/ground_truth/' + f)


    for w in os.listdir('task1/' + use_set + '/mturk_train_data/'):
        for f in os.listdir('task1/' + use_set + '/mturk_train_data/' + w):
            if f not in a:
                os.remove('task1/' + use_set + '/mturk_train_data/' + w + '/' +
                          f)
class baseline_indep:
    """
    aggregate labels ignoring sequence
    """

    def __init__(self, crowdlabs):
        """

        """
        self.crowdlabs = crowdlabs
        self.data = []
        # preprocess:
        for clab in crowdlabs:  # clab = crowd labels for a sentence
            if len(clab) <= 0: continue
            sdata = [([],[]) for i in clab[0].sen ]     # data for this sentence
            for cl in clab:     # a sentence and a worker
                for i, l in enumerate(cl.sen): # word i, label l
                    sdata[i][0].append (cl.wid)
                    sdata[i][1].append (l)

            self.data.extend(sdata)

    def output_gal(self):
        """
        get-another-label format
        """
        f = open('get-another-label/bin/gal_input.txt', 'w')
        for i, x in enumerate(self.data):
            for wid, l in zip(x[0], x[1]):
                f.write(str(wid) + '\t' + str(i) + '\t' + str(l) + '\n')
        f.close()

    def run_gal(self, iterations = 50):
        os.system('get-another-label/bin/get-another-label.sh --iterations %d --categories'
                ' get-another-label/bin/cat.txt --input'
                ' get-another-label/bin/gal_input.txt' % iterations)

    def read_result(self, res_dir = 'get-another-label/bin/results/'):
        fn = res_dir + 'object-probabilities.txt'
        import csv
        f = open(fn)
        reader = csv.DictReader(f, delimiter='\t', quotechar='\"')
        self.l = list(reader)

        self.gal = np.zeros ( (len(self.data, )))
        for dic in self.l:
            i = int (dic['Object'])
            l = int (dic['DS_MaxLikelihood_Category'])
            self.gal[i] = l



    def make_res(self, temp):
        """
        make res from a template of res
        """
        self.res = []
        j = 0
        for i in temp:
            x = self.gal[j: j + len(i)]
            j += len(i)
            self.res.append(x)

def make_sen(sen, res):
    for s, r, in zip(sen, res):
        for i, x in zip(s,r):
            i.label = x

def list_eq(l1, l2):
    if len(l1) != len(l2): return False
    return np.allclose(l1, l2)

def find_res_diff(r1, r2):
    for i, x, y in zip(list(range(len(r1))), r1, r2):
        if not list_eq(x, y):
            print(i, end=' ')
            
            
def write_pico_rod(sid, wid, text_sen, lab):
    f = open('pico_rod/' + str(wid) + '/' + str(sid) + '.txt', 'w')
    for t, l in zip(text_sen, lab):
        f.write(t + ' ' + str(int(l)) + '\n')
    f.close()

def output_pico_rod(cd, features):
    inv_f = {v:k for (k,v) in list(features.items())}

    for sid, sen, clabs in zip(list(range(len(cd.sentences))), cd.sentences, cd.crowdlabs):
        text_sen = []
        for ins in sen:
            text_sen.append(inv_f[ins.features[0]])
        if text_sen == []: continue
        for cl in clabs:
            write_pico_rod(sid, cl.wid, text_sen, cl.sen)
        
def output_gold_pico(gold, l, r, cd, features):
    inv_f = {v:k for (k,v) in list(features.items())}
    
    for i in range(l, r, 1):
        sid = gold[i][0]
        lab = gold[i][1]
        text_sen = []
        sen = cd.sentences[sid]
        for ins in sen:
            text_sen.append(inv_f[ins.features[0]])
        write_pico_rod(sid, 2000, text_sen, lab)
        
        
def to_mace(out_file, cd, n_workers = 47):
    """
    
    """
    f = open(out_file, 'w')
    
    res = []
    wids = []
    for sen, clabs in zip(cd.sentences, cd.crowdlabs):
        #text_sen = get_word_list(sen, features) VERY SLOW
        text_sen = []
        
        n = len(sen)
        for i in range(n):
            labs = ['']* n_workers
            for cl in clabs:
                labs[cl.wid] = str(cl.sen[i])
            for j in range(n_workers):
                f.write(labs[j])
                if j < n_workers - 1:
                    f.write(',')
            f.write('\n')

    f.close()

def read_mace_output(in_file = 'prediction', cd = None, n_workers = 47):
    f = open(in_file)
    res = []
    for sen, clabs in zip(cd.sentences, cd.crowdlabs):
        n = len(sen)
        a = []
        for i in range(n):
            a.append(int(f.readline().strip()))
        res.append(a)
    f.close()
    return res
    
    
def get_pid(ins, inv_label):
    s = inv_label[ins.label]
    a = list(map(int,s.split('_')))
    return a[0]
    
def to_huang(out_file, cd, n_workers = 319, max_line = 1000, indices = None, inv_label = None):
    """
    
    """
    fn = 'huang/'
    abs_cnt = 0
    f = open(fn + '' + str(abs_cnt).zfill(5) + '.txt', 'w')
    f.write('Line\t')
    for x in range(n_workers):
        f.write(str(x) + '\t')
    f.write('\n')
    
    res = []
    wids = []
    line = 0
    last_pid = get_pid(cd.sentences[0][0], inv_label)
    for index, (sen, clabs) in enumerate(zip(cd.sentences, cd.crowdlabs)):
        
        #text_sen = get_word_list(sen, features) VERY SLOW
        text_sen = []
        #if index not in indices: continue
        n = len(sen)
        if n == 0: continue
        if get_pid(sen[0], inv_label) != last_pid:
            last_pid = get_pid(sen[0], inv_label)
            f.close()
            abs_cnt += 1
            f = open(fn + '' + str(abs_cnt).zfill(5) + '.txt', 'w')
            f.write('Line\t')
            for x in range(n_workers):
                f.write(str(x) + '\t')
            f.write('\n')
            
            
        for i in range(n):
            labs = ['0']* n_workers
            for cl in clabs:
                labs[cl.wid] = str(cl.sen[i])
            line += 1
            f.write(str(line) + '\t')
            for j in range(n_workers):
                f.write(labs[j])
                if j < n_workers - 1:
                    f.write('\t')
            f.write('\n')

    f.close()
    
    
def to_huang_ner(out_file, cd, n_workers = 47, max_line = 1000):
    """
    
    """
    fn = 'huang_ner/'
    abs_cnt = 0
    f = open(fn + '' + str(abs_cnt).zfill(5) + '.txt', 'w')
    f.write('Line\t')
    for x in range(n_workers):
        f.write(str(x) + '\t')
    f.write('\n')
    
    res = []
    wids = []
    line = 0
    for index, (sen, clabs) in enumerate(zip(cd.sentences, cd.crowdlabs)):
        
        #text_sen = get_word_list(sen, features) VERY SLOW
        text_sen = []
        #if index not in indices: continue
        n = len(sen)
        if n == 0: continue
        if (index+1) % 20 == 0:
            f.close()
            abs_cnt += 1
            f = open(fn + '' + str(abs_cnt).zfill(5) + '.txt', 'w')
            f.write('Line\t')
            for x in range(n_workers):
                f.write(str(x) + '\t')
            f.write('\n')
            
            
        for i in range(n):
            labs = ['0']* n_workers
            for cl in clabs:
                # 9 is non entity 
                labs[cl.wid] = str(0 if cl.sen[i] == 9 else 1)
            line += 1
            f.write(str(line) + '\t')
            for j in range(n_workers):
                f.write(labs[j])
                if j < n_workers - 1:
                    f.write('\t')
            f.write('\n')

    f.close()


def data_to_hmm_crowd_format(annotations, text, doc_start, outputdir, overwrite=False):

    filename = outputdir + '/hmm_crowd_text_data.pkl'

    if not os.path.exists(filename) or overwrite:
        if text is not None:

            ufeats = {}
            features = np.zeros(len(text))
            for t, tok in enumerate(text.flatten()):

                if np.mod(t, 10000) == 0:
                    print('converting data to hmm format: token %i / %i' % (t, len(text)))

                if tok not in ufeats:
                    ufeats[tok] = len(ufeats)

                features[t] = ufeats[tok]

            features = features.astype(int)

        else:
            features = []
            ufeats = []

        sentences_inst = []
        crowd_labels = []

        for i in range(annotations.shape[0]):

            if len(features):
                token_feature_vector = [features[i]]
            else:
                token_feature_vector = []

            label = 0

            if doc_start[i]:
                sentence_inst = []
                sentence_inst.append(instance(token_feature_vector, label + 1))
                sentences_inst.append(sentence_inst)

                crowd_labs = []
                for worker in range(annotations.shape[1]):

                    if annotations[i, worker] == -1: # worker has not seen this doc
                        continue

                    worker_labs = crowdlab(worker, len(sentences_inst) - 1, [int(annotations[i, worker])])

                    crowd_labs.append(worker_labs)
                crowd_labels.append(crowd_labs)

            else:
                sentence_inst.append(instance(token_feature_vector, label + 1))


                for widx, worker_labs in enumerate(crowd_labs):
                    crowd_labs[widx].sen.append(int(annotations[i, worker_labs.wid]))

        nfeats = len(ufeats)

        print('writing HMM pickle file...')
        with open(filename, 'wb') as fh:
            pickle.dump([sentences_inst, crowd_labels, nfeats], fh)
    else:
        with open(filename, 'rb') as fh:
            data = pickle.load(fh)
        sentences_inst = data[0]
        crowd_labels = data[1]
        nfeats = data[2]

    return sentences_inst, crowd_labels, nfeats

def subset_hmm_crowd_data(sentences_inst, crowd_labels, docsubsetidxs, nselected_by_doc):

    if docsubsetidxs is not None:
        sentences_inst = np.array(sentences_inst)
        sentences_inst = sentences_inst[docsubsetidxs]

        crowd_labels = [crowd_labels[d] for d in docsubsetidxs]

        for d, crowd_labels_d in enumerate(crowd_labels):
            crowd_labels[d] = crowd_labels_d[:int(nselected_by_doc[d])]
        crowd_labels = np.array(crowd_labels)

    if len(sentences_inst[0][0].features) and isinstance(sentences_inst[0][0].features[0], float):
        # the features need to be ints not floats!
        for s, sen in enumerate(sentences_inst):
            for t, tok in enumerate(sen):
                feat_vec = []
                for f in range(len(tok.features)):
                    feat_vec.append(int(sentences_inst[s][t] .features[f]))
                sentences_inst[s][t] = instance(feat_vec, sentences_inst[s][t].label)

    return sentences_inst, crowd_labels