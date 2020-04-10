import numpy as np
from baselines import util
import os
import copy
import nltk
#import crf
import scipy.special
import sklearn

class HMM:
    """
    Hidden Markov Model
    """

    def __init__(self, n, m):
        """
        fix n, m
        :param n: number of states
        :param m: number of observations
        """
        self.n = n
        self.m = m
        self.t = np.zeros((n, n))
        self.e = np.zeros((n, m))
        self.start = np.asarray([1.0 / n] * n)

    def pr_obs(self, i, list_features, t=None):
        """
        :param i: state
        :param list_features:
        :param t: time, not used here
        :return: probability of observing the features in state i
        """
        res = 1
        for f in list_features:
            res *= self.e[i, f]
        return res

    def decode(self, a, include_crowd_obs=False):
        """
        Viterbi decoding
        :param a: seq of observations, each observation is a list of features
        :return:
        """
        l = len(a)
        if l == 0:
            return []
        # c[t][i] = prob of best path time t, at state i
        c = np.zeros((l, self.n))
        c[0] = np.copy(self.start)   # * self.e[:, a[0]]
        # print self.n, c.shape
        for i in range(self.n):
            c[0][i] *= self.pr_obs(i, a[0])

        # b[t][i] = backpointer
        b = np.zeros((l, self.n))

        for t in range(1, l, 1):  # time
            ob = a[t]

            for i in range(self.n):  # current state
                for j in range(self.n):  # previous state
                    # todo: change to log scale
                    p = c[t - 1][j] * self.t[j, i] * self.pr_obs(i, ob)
                    if include_crowd_obs:
                        p *= self.pr_crowd_labs(t, i, self.current_list_cl)
                    # print t, i, j, p
                    if p > c[t][i]:
                        c[t][i] = p
                        b[t][i] = j

            # normalise otherwise p ends up as zeros with long sequences
            c_t_total = 0
            for i in range(self.n):
                c_t_total += c[t][i]

            for i in range(self.n):
                c[t][i] /= c_t_total

        res = np.zeros((l,))

        # trace
        p = 0
        for i in range(self.n):
            if c[l - 1][i] > p:
                p = c[l - 1][i]
                res[l - 1] = i

        seq_prob = p

        for t in range(l - 2, -1, -1):
            res[t] = b[int(t + 1), int(res[t + 1])]

        # print c
        # print b
        return res, seq_prob

    def learn(self, sentences, smooth=0.001):
        """
        learn parameters from labeled data
        :param sentences: list of sentence, which is list of instance
        :return:
        """
        # counting
        self.t = smooth * np.ones((self.n, self.n))
        self.e = smooth * np.ones((self.n, self.m))
        self.start = smooth * np.ones((self.n,))

        for sentence in sentences:
            if len(sentence) > 0:
                i = sentence[0]
                self.start[i.label] += 1
                prev = -1  # previous state
                for i in sentence:
                    state = i.label
                    if prev != -1:
                        self.t[prev][state] += 1
                    for f in i.features:
                        self.e[state][int(f)] += 1
                    prev = state

        # save count for e
        self.count_e = copy.deepcopy(self.e)

        # normalizing
        self.start = self.start * 1.0 / np.sum(self.start)
        for i in range(self.n):
            self.t[i] = self.t[i] * 1.0 / np.sum(self.t[i])
            self.e[i] = self.e[i] * 1.0 / np.sum(self.e[i])

    def decode_all(self, sentences):
        self.res = []
        self.res_prob = []
        for s in sentences:
            mls, mls_prob = self.decode(util.get_obs(s))
            self.res.append(mls)
            self.res_prob.append(mls_prob)

##########################################################################
##########################################################################
##########################################################################
##########################################################################


class WorkerModel:
    """
    model of workers
    """
    def __init__(self, n_workers = 47, n_class = 10, smooth = 0.001, ne = 9, rep = 'cv'):
        """

        :param n_workers:
        :param n_class:
        :param smooth:
        :param ne:
        :param rep: representation. cv2 = confusion vec of accuracy in two cases: non-entity/ entity
        """
        self.n_workers = n_workers
        self.n = n_class
        self.smooth = smooth
        self.ne = ne
        self.rep = rep

    def learn_from_pos(self, data, pos):
        """

        :param data: crowd_data
        :param pos: sentence posterior
        :return:
        """
        count = self.smooth * np.ones( (self.n_workers, self.n, self.n))
        for i, sentence in enumerate(data.sentences):
            for j in range(len(sentence)):
                for l, w in data.get_lw(i, j):
                    for k in range(self.n):  # 'true' label = k
                        count[w][k][l] += pos[i][j][k]
        self.learn_from_count(count)

    def learn_from_count(self, count):
        """

        :return:
        """
        #save the count for debug
        self.count = count

        if self.rep == 'cv2':
            ne = self.ne
            self.cv = np.zeros((self.n_workers, 2))
            for w in range(self.n_workers):
                self.cv[w][0] = count[w][ne][ne] * 1.0 / np.sum(count[w][ne]) # accuracy for ne class

                cc = self.smooth; cw = self.smooth # count for correct and wrong for non ne classes
                for i in range(self.n):
                    if i != ne:
                        cc += count[w][i][i]
                        cw += np.sum(count[w][i]) - count[w][i][i]
                self.cv[w][1] = cc * 1.0 / (cc + cw)
        elif self.rep == 'cv':
            self.cv = np.zeros((self.n_workers, self.n))
            for w in range(self.n_workers):

                if np.mod(w, 100) == 0:
                    print('M-step, processing worker counts %i of %i' % (w, self.n_workers))

                for i in range(self.n):
                    self.cv[w][i] = count[w][i][i] * 1.0 / np.sum(count[w][i]) # accuracy for ne class

        elif self.rep == 'cm_sage':
            self.cm = np.zeros((self.n_workers, self.n, self.n))
            # background dist
            m = np.sum(count, axis=0)
            for i in range(self.n): m[i] = m[i] * 1.0 / np.sum(m[i])
            m = np.log(m)

            for w in range(self.n_workers):
                for i in range(self.n):
                    temp = additive.estimate(count[w][i], m[i])
                    temp = np.reshape(temp, (self.n,) )
                    self.cm[w][i] = np.exp(temp + m[i])
                    self.cm[w][i] = self.cm[w][i] * 1.0 / np.sum(self.cm[w][i])

        else:
            self.cm = np.zeros((self.n_workers, self.n, self.n))
            for w in range(self.n_workers):
                for k in range(self.n):
                    self.cm[w][k] = count[w][k] * 1.0 / np.sum(count[w][k])




    def get_prob(self, w, true_lab, lab):
        """

        :param w: worker
        :param true_lab:
        :param lab:
        :return: probability of response lab given true label
        """
        #return 1.0
        if self.rep == 'cv2':
            if self.ne == true_lab:
                if true_lab == lab:
                    return self.cv[w][0]
                else:
                    return (1 - self.cv[w][0]) / float(self.n - 1)
            else:
                if true_lab == lab:
                    return self.cv[w][1]
                else:
                    return (1 - self.cv[w][1]) / float(self.n - 1)

        elif self.rep == 'cv':
            if true_lab == lab:
                return self.cv[w][true_lab]
            else:
                return (1 - self.cv[w][true_lab]) / float(self.n - 1)
        elif self.rep == 'cm_sage':
            return self.cm[w][true_lab][lab]
        else:
            return self.cm[w][true_lab][lab]



class HMM_crowd(HMM):

    def __init__(self, n, m, data, features, labels, n_workers=47, init_w=0.9, smooth=0.001, smooth_w=10, ne = 9, vb = None):
        """
        :param data: util.crowd_data with crowd label
        :return:
        """
        HMM.__init__(self, n, m)
        self.data = data
        self.smooth = smooth
        self.n_workers = n_workers
        self.ep = 1e-300
        self.features = features
        self.labels = labels
        self.init_w = init_w
        self.ne = ne

        #self.wsen = np.zeros((n_workers,))
        #self.wspe = np.zeros((n_workers,))
        self.wca = np.zeros((n, n_workers))
        #self.ne = labels['O']  # id of 'non entity' label
        self.ne = ne
        self.smooth_w = smooth_w
        self.n_sens = len(data.sentences)
        self.vb = vb

    def pr_crowd_labs(self, t, i, list_cl):
        """
        :param t: time
        :param i: the state
        :param list_cl: list of util.crowddlab
        :return: probability of observing crowd labels at state i
        """
        res = 1# * self.prior[i]
        for cl in list_cl:
            wid = cl.wid
            sen = cl.sen
            lab = sen[t]                        # crowd label
            # if i == self.ne:
            #    res *= self.wspe[wid] if lab == i else 1 - self.wspe[wid] # specificity
            # else:
            # res *= self.wsen[wid] if lab == i else 1 - self.wsen[wid] #
            # sensitivity
            #res *= self.wca[i, wid] if lab == i else 1 - self.wca[i, wid]
            #res *= self.wa[wid][i][lab]
            res *= self.wm.get_prob(wid, i, lab)

        return res

    def inference(self, sentence, list_cl, return_ab=False):
        T = len(sentence)                   # number of timesteps
        alpha = np.zeros((T, self.n))  # T * states
        beta = np.zeros((T, self.n))

        # alpha (forward):
        for i in range(self.n):
            alpha[0][i] = self.pr_obs(
                i, sentence[0].features) * self.pr_crowd_labs(0, i, list_cl) * self.start[i]

        for t in range(1, T, 1):
            ins = sentence[t]

            alpha_t_sum = 0

            for i in range(self.n):             # current state
                alpha[t][i] = 0
                for j in range(self.n):         # previous state
                    alpha[t][i] += self.pr_obs(i, ins.features) * self.t[j][i] * alpha[t - 1][j] \
                        * self.pr_crowd_labs(t, i, list_cl)

                alpha_t_sum += alpha[t][i]

            # normalise
            for i in range(self.n):
                alpha[t][i] /= alpha_t_sum

        # beta (backward):
        for i in range(self.n):
            beta[T - 1][i] = self.pr_obs(i, sentence[T - 1].features) * \
                self.pr_crowd_labs(T - 1, i, list_cl)

        for t in range(T - 2, -1, -1):
            ins = sentence[t + 1]

            beta_t_sum = 0

            for i in range(self.n):             # current state
                beta[t][i] = 0
                for j in range(self.n):         # next state
                    beta[t][i] += self.pr_obs(j, ins.features) * self.t[i][j] * beta[t + 1][j] \
                        * self.pr_crowd_labs(t + 1, j, list_cl)#\
                        #* (self.start[i] if t == 0 else 1)

                beta_t_sum += beta[t][i]

            for i in range(self.n):
                beta[t][i] /= beta_t_sum

        if return_ab:
            return (alpha, beta)

        sen_posterior = []
        # update counts
        p = np.zeros((self.n,))
                
        for t in range(T):
            for i in range(self.n):
                p[i] = self.ep + alpha[t][i] * beta[t][i]
            p = p * 1.0 / np.sum(p)  # normalilze

            #save the posterior
            sen_posterior.append(p.copy())

            if t == 0:  # update start counts
                self.count_start += p

            # update prior count
            #self.count_prior += p

            # update emission counts
            ins = sentence[t]
            for i in range(self.n):
                for f in ins.features:
                    self.count_e[i][f] += p[i]

            # update crowd params counts
            for i in range(self.n):                                 # state
                for cl in list_cl:
                    wid = cl.wid
                    # worker ans
                    lab = cl.sen[t]
                    # if i == self.ne:
                    #    if lab == self.ne:
                    #        self.count_spe[wid][0] += p[i]
                    #    else:
                    #        self.count_spe[wid][1] += p[i]
                    # else:
                    #    if lab == self.ne:
                    #        self.count_sen[wid][0] += p[i]
                    #    else:
                    #        self.count_sen[wid][1] += p[i]
                    #if lab == i:
                    #    self.count_wa[i, wid][1] += p[i]
                    #else:
                    #    self.count_wa[i, wid][0] += p[i]
                    self.count_wa[wid][i][lab] += p[i]



        trans_pos = []
        # update transition counts
        for t in range(T - 1):
            p = np.zeros((self.n, self.n))
            ins = sentence[t+1]
            for i in range(self.n):         # state at time t
                for j in range(self.n):     # state at time t+1
                    p[i][j] = self.ep + alpha[t][i] * self.t[i][j] * self.pr_obs(j, ins.features) \
                        * self.pr_crowd_labs(t + 1, j, list_cl) * beta[t + 1][j]

            # update transition counts
            p = p * 1.0 / np.sum(p)
            for i in range(self.n):
                #p[i] = p[i] * 1.0 / np.sum(p[i])
                self.count_t[i] += p[i]

            trans_pos.append(p.copy())

        # log likelihood
        ll = np.log( np.sum(alpha[t-1]) )
        
        return (sen_posterior, trans_pos, ll)

    def e_step(self):
        """
        do alpha-beta passes
        :return:
        """
        # setup counting
        self.count_t = self.smooth * np.ones((self.n, self.n))
        self.count_e = self.smooth * np.ones((self.n, self.m))
        self.count_start = self.smooth * np.ones((self.n,))
        #self.count_sen = self.smooth * np.ones((self.n_workers,2))
        #self.count_spe = self.smooth * np.ones((self.n_workers, 2))
        #self.count_wca = self.smooth_w * np.ones((self.n, self.n_workers, 2))        
        #self.count_prior = self.smooth * np.ones( (self.n,) )
        self.count_wa = self.smooth * np.ones( (self.n_workers, self.n, self.n) )


        self.sen_posterior = []
        self.trans_posterior = []
        sum_ll = 0
        for i, sentence in enumerate(self.data.sentences):

            if np.mod(i, 100) == 0:
                print('E-step, processing sentence %i of %i' % (i, len(self.data.sentences)))

            if len(sentence) > 0:
                sen_pos, trans_pos, ll = self.inference(sentence, self.data.crowdlabs[i])
                sum_ll += ll
            else:
                sen_pos, trans_pos = ([], [])

            self.sen_posterior.append (sen_pos)
            self.trans_posterior.append(trans_pos)

        # save sum of log-likelihood
        self.sum_ll = sum_ll

    def m_step(self):
        if self.vb != None:
            self.m_step_vb()
            return
        # normalize all the counts
        self.start = self.count_start * 1.0 / np.sum(self.count_start)
        #self.prior = self.count_prior * 1.0 / np.sum(self.count_prior)

        for i in range(self.n):
            if np.mod(i, 100) == 0:
                print('M-step, processing class counts %i of %i' % (i, self.n))
            self.t[i] = self.count_t[i] * 1.0 / np.sum(self.count_t[i])
            self.e[i] = self.count_e[i] * 1.0 / np.sum(self.count_e[i])

        self.wm.learn_from_count(self.count_wa)
        #for w in range(self.n_workers):
            #self.wsen[w] = self.count_sen[w][1] * 1.0 / (self.count_sen[w][0] + self.count_sen[w][1])
            #self.wspe[w] = self.count_spe[w][0] * 1.0 / (self.count_spe[w][0] + self.count_spe[w][1])
            #ne = self.ne
            #self.wca[ne, w] = self.count_wca[ne, w][1] * 1.0 / \
            #    (self.count_wca[ne, w][0] + self.count_wca[ne, w][1])
            # sum over pos class (not include non-entity)
            #sum_pos_1 = np.sum(
            #    self.count_wca[:, w, 1]) - self.count_wca[ne, w, 1]
            #sum_pos_0 = np.sum(
            #    self.count_wca[:, w, 0]) - self.count_wca[ne, w, 0]
            #for i in range(self.n):
            #    if i != ne:
                    #self.wca[i, w] = self.count_wca[i, w][1] * 1.0 / (self.count_wca[i, w][0] + self.count_wca[i, w][1])
            #        self.wca[i, w] = sum_pos_1 * 1.0 / (sum_pos_1 + sum_pos_0)
            #for i in range(self.n):
            #    self.wa[w][i] = self.count_wa[w][i] * 1.0 / np.sum(self.count_wa[w][i])
            #self.wm.learn_from_pos(self.data, self.sen_posterior)

    def m_step_vb(self):
        """
        use Variational Bayes
        """
        self.start = self.count_start * 1.0 / np.sum(self.count_start)
        f = lambda x: np.exp( scipy.special.digamma(x))

        for i in range(self.n):
            if np.mod(i, 100) == 0:
                print('VB M-step, processing class counts %i of %i' % (i, self.n))

            self.count_t[i] = self.count_t[i] - self.smooth + self.vb[0]
            self.count_e[i] = self.count_e[i] - self.smooth + self.vb[1]

            self.t[i] = f(self.count_t[i] * 1.0) / f(np.sum(self.count_t[i]))
            self.e[i] = f(self.count_e[i] * 1.0) / f(np.sum(self.count_e[i]))

        self.wm.learn_from_count(self.count_wa)

    def init_te_from_pos(self, pos):
        """
        init transition and emission from posterior
        """
        self.t = self.smooth * np.ones((self.n, self.n))
        self.e = self.smooth * np.ones((self.n, self.m))
        self.start = self.smooth * np.ones((self.n,))

        for sentence, p in zip(self.data.sentences, pos):
            if len(sentence) > 0:
                self.start += p[0]
                for t, ins in enumerate(sentence):
                    for i in range(self.n): #current state
                        for f in ins.features:
                            self.e[i][f] += p[t][i]
                        if t > 0:
                            for j in range(self.n): #previous state
                                self.t[j][i] += p[t-1][j] * p[t][i]



        # normalizing
        self.start = self.start * 1.0 / np.sum(self.start)
        for i in range(self.n):
            self.t[i] = self.t[i] * 1.0 / np.sum(self.t[i])
            self.e[i] = self.e[i] * 1.0 / np.sum(self.e[i])


    def init(self, init_type='mv', sen_a=1, sen_b=1, spe_a=1, spe_b=1, wm_rep =
            'cv', save_count_e = False, dw_em = 5, wm_smooth = 0.001):
        """

        :param init_type:

        :param sen_a:  :param sen_b: :param spe_a: :param spe_b: priors for sen, spe
        expect MV to over-estimate worker
        :return:
        """
        if init_type == 'mv':
            h = HMM(self.n, self.m)
            mv_sen = util.mv_cd(self.data, self.labels)
            h.learn(mv_sen, smooth=self.smooth)

            for i in range(self.n):
                self.start[i] = h.start[i]

            #(correct_0, correct_1, wrong_0, wrong_1) = util.cal_workers_true_acc(self.data, util.get_all_lab(mv_sen), ne = self.ne)
            self.wca = util.cal_workers_true_acc(
                self.data, util.get_all_lab(mv_sen), ne=self.ne, return_ss=True)

            # for i in range(self.n_workers):
            #self.wsen[i] = (correct_1[i] + sen_a) * 1.0 / (wrong_1[i] + correct_1[i] + sen_a + sen_b)
            #self.wsen[i] = 0.5
            #self.wspe[i] = (correct_0[i] + spe_a) * 1.0 / (wrong_0[i] + correct_0[i] + spe_a + spe_b)

            for s in range(self.n):
                for s2 in range(self.n):
                    self.t[s][s2] = h.t[s][s2]
                    # for s in range(self.n):
                for o in range(self.m):
                    self.e[s][o] = h.e[s][o]
            #save the count of e for hmm_sage
            if save_count_e:
                self.count_e = h.count_e
            #self.init2()
        elif init_type == 'dw':
            d = dw(self.n, self.m, self.data, self.features, self.labels,
                           self.n_workers, self.init_w, self.smooth)
            d.init()
            d.em(dw_em)
            d.mls()
            self.d = d

            h = HMM(self.n, self.m)
            sen = copy.deepcopy(self.data.sentences)
            util.make_sen(sen, d.res)
            h.learn(sen, smooth = self.smooth)
            #self.wm = WorkerModel()
            #self.wm.learn_from_pos(self.data, d.pos)

            self.wm = WorkerModel(n_workers = self.n_workers, n_class = self.n,
                    rep=wm_rep, ne = self.ne, smooth = wm_smooth)
            self.wm.learn_from_pos(self.data, d.pos)

            self.start = h.start
            for s in range(self.n):
                for s2 in range(self.n):
                    self.t[s][s2] = h.t[s][s2]
                    # for s in range(self.n):
                for o in range(self.m):
                    self.e[s][o] = h.e[s][o]

            #self.init_te_from_pos(d.pos)
            #save the count of e for sage
            if save_count_e:
                self.count_e = h.count_e

            self.h = h

        else:
            # init params (uniform)
            for i in range(self.n):
                self.start[i] = 1.0 / self.n
                self.wa = [0.9] * self.n_workers
            for s in range(self.n):
                for s2 in range(self.n):
                    self.t[s][s2] = 1.0 / self.n

            for s in range(self.n):
                for o in range(self.m):
                    self.e[s][o] = 1.0 / self.m

            for w in range(self.n_workers):
                #self.wsen[i] = 0.9
                #self.wspe[i] = 0.6
                for i in range(self.n):
                    self.wca[i, w] = 0.8


    def init2(self):
        """
        init
        """
        pos = []
        self.prior = np.zeros( (self.n,) )

        for i, sentence in enumerate(self.data.sentences):
            pos.append( self.smooth * np.ones((len(sentence), self.n)) )
            for j in range(len(sentence)):
                for l in self.data.get_labs(i, j): # labels for sen i, pos j
                    pos[i][j][l] += 1
                pos[i][j] = pos[i][j] * 1.0 / np.sum(pos[i][j])
                self.prior += pos[i][j]

        self.prior = self.prior * 1.0 / np.sum(self.prior)


    
    def learn(self, num=4):
        """
        learn by EM
        :return:
        """
        # init params (uniform)
        # for i in range(self.n):
        #     self.start[i] = 1.0/self.n
        # self.wa = [0.9] * self.n_workers
        # for s in range(self.n):
        #     for s2 in range(self.n):
        #         self.t[s][s2] = 1.0/self.n
        #
        # for s in range(self.n):
        #     for o in range(self.m):
        #         self.e[s][o] = 1.0/self.m
        self.init()
        self.em(num)

    def em(self, num=4):
        # run EM
        for it in range(num):
            print('HMM-crowd running e-step %i of %i' % (it, num))
            self.e_step()

            print('HMM-crowd running m-step %i of %i' % (it, num))
            self.m_step()

    def mls(self):
        """
        compute the most likely states seq for all sentences
        :return:
        """
        self.res = []
        self.res_prob = []
        for s, sentence in enumerate(self.data.sentences):
            if len(sentence) > 0:
                self.current_list_cl = self.data.crowdlabs[s]
                ml_states, ml_probs = self.decode(util.get_obs(
                    sentence), include_crowd_obs=True)
                self.res.append(ml_states)
                self.res_prob.append(ml_probs)
            else:
                self.res.append([])
                self.res_prob.append(1)

    def marginal_decode(self, th):
        """
        decode by marginal prob
        """
        self.res = []
        for i in range(len(self.data.sentences)):
            temp = []
            for j in range(len(self.sen_posterior[i])):
                temp.append ( np.argmax(self.sen_posterior[i][j]) )
            self.res.append(temp)


    def decode_sen_no(self, s):
        self.current_list_cl = self.data.crowdlabs[s]
        sentence = self.data.sentences[s]

        ml_states, ml_probs = self.decode(util.get_obs(
            sentence), include_crowd_obs=True)

        return ml_states


    def threshold(self, thresh = 0.9):
        self.flag = np.zeros((self.n_sens,), dtype=bool)
        for i, r in enumerate(self.res):
            for j, l in enumerate(r):
                if self.posterior[i][j][int(l)] < thresh:
                    self.flag[i] = True

def pos_decode(pos, th, ne = 9):
        """
        decode by posterior:
        res = argmax if pro > th
        else res = ne
        """
        res = []
        for i in range(len(pos)):
            temp = []
            for j in range(len(pos[i])):                
                #p = copy.copy(pos[i][j])
                #p[ne] = -1
                k = np.argmax(pos[i][j])
                if pos[i][j][k] > th:
                    temp.append(k)
                else:
                    temp.append(ne)
            res.append(temp)
        return res

##########################################################################
##########################################################################


class HMM_sage(HMM_crowd):

    def __init__(self, n, m, data, features, labels, n_workers=47, init_w=0.9, smooth=0.001, smooth_w=10):
        HMM_crowd.__init__(self, n, m, data, features, labels,
                           n_workers, init_w, smooth)
        HMM.eta = np.zeros((self.m, self.n))

    def init(self, init_type = 'dw', wm_rep = 'cm'):
        HMM_crowd.init(self, init_type=init_type, wm_rep=wm_rep, save_count_e = True)
        self.estimate_sage()

    def estimate_sage(self, mult = 2.0):
        # dont do sage for non-entity
        self.count_e[self.ne, :] = np.zeros((self.m,))
        eq_m = np.sum(self.count_e, axis=0) / np.sum(self.count_e)
        #eq_m = 1.0 / self.m * np.ones((self.m))
        eq_m = np.log(eq_m)

        eta = additive.estimate(mult*self.count_e.T, eq_m)
        for i in range(self.n):
            if i != self.ne:
                self.e[i] = np.exp(eta[:, i] + eq_m) * 1.0 / \
                    np.sum(np.exp(eta[:, i] + eq_m))

        # save eq_m and eta
        self.eq_m = eq_m
        self.eta = eta

    def m_step(self):
        HMM_crowd.m_step(self)
        self.estimate_sage()




class dw(HMM_crowd):
    """
    """
    def __init__(self, n, m, data, features, labels, n_workers=47, init_w=0.9, smooth=0.001, smooth_w=10):
        """
        n: number of states
        :param data: util.crowd_data with crowd label
        :return:
        """
        HMM_crowd.__init__(self, n, m, data, features, labels,
                           n_workers, init_w, smooth)
 

    def init(self):
        self.pos = []
        self.prior = np.zeros( (self.n,) )

        for i, sentence in enumerate(self.data.sentences):
            self.pos.append( self.smooth * np.ones((len(sentence), self.n)) )
            for j in range(len(sentence)):
                for l in self.data.get_labs(i, j): # labels for sen i, pos j
                    self.pos[i][j][l] += 1
                self.pos[i][j] = self.pos[i][j] * 1.0 / np.sum(self.pos[i][j])
                self.prior += self.pos[i][j]

        self.prior = self.prior * 1.0 / np.sum(self.prior)


    def e_step(self):
        for i, sentence in enumerate(self.data.sentences):
            if np.mod(i, 100) == 0:
                print('dw e-step, sentence %i of %i' % (i, len(self.data.sentences)))
            self.pos[i] = np.ones( (len(sentence), self.n) )
            for j in range(len(sentence)):
                self.pos[i][j] = self.prior.copy()
                for l, w in self.data.get_lw(i, j): # labels for sen i, pos j
                    self.pos[i][j] *= self.wa[w][:,l]
                self.pos[i][j] = self.pos[i][j] * 1.0 / np.sum(self.pos[i][j])
 

    def m_step(self):        
        count = self.smooth * np.ones ( (self.n_workers, self.n, self.n) )
        count_prior = self.smooth * np.ones_like(self.prior)
        #get-another-label heuristic: 0.9 to diagonal, uniform to elsewhere

        print('dw m-step')
        for w in range(self.n_workers):
            #print('dw m-step, worker %i of %i' % (w, self.n_workers))

            for i in range(self.n):
                for j in range(self.n):
                    count[w][i][j] = 0.9 if i == j else 0.1 / (self.n-1)

        for i, sentence in enumerate(self.data.sentences):
            #print('dw w-step, sentence %i of %i' % (i, len(self.data.sentences)))

            for j in range(len(sentence)):
                count_prior += self.pos[i][j]
                for l, w in self.data.get_lw(i,j):
                    for k in range(self.n): # 'true' label = k
                        count[w][k][l] += self.pos[i][j][k]

        self.prior = count_prior * 1.0 / np.sum(count_prior)

        self.wa = np.zeros( (self.n_workers, self.n, self.n) )
        for w in range(self.n_workers):
            #print('dw m-step part 3, worker %i of %i' % (w, self.n_workers))

            for k in range(self.n):
                self.wa[w][k] = count[w][k] * 1.0 / np.sum(count[w][k])


    def em(self, iterations = 3):
        self.init()
        self.m_step()

        for it in range(iterations):
            self.e_step()
            self.m_step()

    def mls(self):
        self.res = []
        for i, sentence in enumerate(self.data.sentences):
            self.res.append([0] * len(sentence))
            for j in range(len(sentence)):
                self.res[i][j] = np.argmax(self.pos[i][j])


##########################################################################
##########################################################################
data_atis = 'atis3_features.txt'


def run_test(test, h):
    cnt = 0
    correct = 0
    for s in test:
        x = util.get_obs(s)
        g = util.get_lab(s)
        p = h.decode(x)
        for i, j in zip(g, p):
            cnt += 1
            if i == j:
                correct += 1

    print(correct * 1.0 / cnt)


def run(smooth, filename=data_atis):
    #train, test, features, labels = util.load(filename)
    #train, test, features, labels = util.load('data/WSJ_all.txt', 0.49)
    train, test, features, labels = util.load('atis3_features.txt')
    n = len(labels) + 1
    m = len(features) + 1
    h = HMM(n, m)

    h.learn(train, smooth)
    run_test(test, h)

    return h


def run_crowd(filename='atis3_features.txt'):
    train, test, features, labels = util.load(filename)
    n = len(labels) + 1
    m = len(features) + 1
    s = util.simulator(train, features, labels, 1)
    s.simulate()
    hc = HMM_crowd(n, m, s.cd, features, labels)
    return hc


def split_rod(all_sen, cd, features, labels):
    """
    split rod data into validation/test
    """
    n = len(all_sen) / 2
    sen_val = all_sen[:n]
    sen_test = all_sen[n:]

    cd_val = util.crowd_data(sen_val, cd.crowdlabs[:n])
    cd_test = util.crowd_data(sen_test, cd.crowdlabs[n:])

    return (sen_val, sen_test, cd_val, cd_test)


def run_rod(smooth = 0.001, dirname = 'task1/val/'):
    """
    use_set: validation or test
    """
    #if use_set == 'val':
    #    dirname = 'task1/val/'
    #elif use_set == 'test':
    #    dirname = 'task1/test/'
    #else:
    #    dirname = ''

    # read ground truth
    all_sen, features, labels, docs = util.read_rod(dirname = dirname +
        'ground_truth')
    # read crowd labels
    cd = util.read_workers_rod(all_sen, features, labels, docs, dirname =
            dirname + 'mturk_train_data')

    n = len(labels) + 1
    m = len(features) + 1
     
    hc = HMM_crowd(n, m, cd, features, labels, smooth = smooth)
    hs = HMM_sage(n, m, cd, features, labels, smooth = smooth)
    return hs, hc, all_sen, features, labels


def list_entities(sen, st, inside):
    """
    list the occurence of an entity
    """
    n = len(sen)
    res = []
    i = 0
    while i < n:
        if sen[i] == st:
            x = i
            i += 1
            while i < n and sen[i] == inside:
                i += 1
            res.append((x, i - 1))
        else:
            i += 1
    return res


def eval_ner(gold, sen, labels):
    """
    evaluate NER
    """

    if len(gold) != len(sen):
        print(len(gold), len(sen))
        raise "lenghts not equal"

    tp = 0
    #tn = 0
    fp = 0
    fn = 0

    list_en = ["LOC", "MISC", "ORG", "PER"]
    for en in list_en:
        g = list_entities(gold, labels["B-" + en], labels["I-" + en])
        s = list_entities(sen, labels["B-" + en], labels["I-" + en])
        for loc in g:
            if loc in s:
                tp += 1
            else:
                fn += 1

        for loc in s:
            if loc not in g:
                fp += 1

    return (tp, fp, fn)


def eval_hc_train(hc, labels, print_err = False):
    """
    evaluate in the train set
    :param hc:
    :param labels:
    :return:
    """
    n = len(hc.res)
    tp = 0
    fp = 0
    fn = 0

    for i in range(n):
        if len(hc.data.sentences[i]) > 0:
            (x, y, z) = eval_ner(util.get_lab(
                hc.data.sentences[i]), hc.res[i], labels)
            tp += x
            fp += y
            fn += z



    try:
        pre = tp * 1.0 / (tp + fp)
        rec = tp * 1.0 / (tp + fn)
        f = 2.0 * pre * rec / (pre + rec)
        print(pre, rec, f)
    except ZeroDivisionError:
        print("DIV BY 0 ", tp, fp, fn)


def has_oov(sen):
    for i in sen:
        for j in i.features:
            if j == 0:
                return True
    return False


def get_tag_hc(hc, sen):
    return hc.decode(util.get_obs(sen))

def get_tag_t(tagger, sen, features):
    words = []
    for i in sen:
        words.append(i.word)
    x = nltk.pos_tag(words)
    x = [crf.word2features(x, i) for i in range(len(x))]
    tags = tagger.tag(x)
    return list(map(int, tags))


def get_tag(f, sen, features, decoder):
    if decoder == 'hc':
        return get_tag_hc(f, sen)
    else:
        return get_tag_t(f, sen, features)

def eval_hc_test(hc, features, labels, print_err=False, decoder='hc'):
    """
    evaluate in the train set
    :param hc:
    :param labels:
    :return:
    """
    tp = 0
    fp = 0
    fn = 0

    dirname = "testa"
    input = []
    for file in os.listdir(dirname):
        # print file
        if file.endswith(".txt"):
            f = open(os.path.join(dirname, file))
            l = list(f)
            input.extend(l)
            f.close()
    # return input
    sentences = util.extract(input, features, labels, keep_word = True)

    # return sentences

    for sen in sentences:
        if True:
            # if not has_oov(sen):
            #predicted = hc.decode(util.get_obs(sen))
            predicted = get_tag(hc, sen, features, decoder)
            (x, y, z) = eval_ner(util.get_lab(sen), predicted, labels)
            tp += x
            fp += y
            fn += z
            if print_err:
                if y + z > 0:
                    print("sen: ",  util.get_words(sen, features) + " OOV = " + str(has_oov(sen)))
                    print("true labels: ", util.get_lab_name(util.get_lab(sen), labels))
                    print("predicted: ", util.get_lab_name(predicted, labels))

    try:
        pre = tp * 1.0 / (tp + fp)
        rec = tp * 1.0 / (tp + fn)
        f = 2.0 * pre * rec / (pre + rec)
        print(pre, rec, f)
    except ZeroDivisionError:
        print("DIV BY 0 ", tp, fp, fn)


def eval_seq_train(gold, pre, labels, hc = None, features = None):
    """
    evaluate a sequence labeler
    """
    n = len(gold)
    tp = 0
    fp = 0
    fn = 0

    for i in range(n):
        (x, y, z) = eval_ner(gold[i], pre[i], labels)
        tp += x
        fp += y
        fn += z
        if hc != None:
            if y + z > 0:
                sen = hc.sentences[i]
                print("sen: ",  util.get_words(sen, features) + " OOV = " + str(has_oov(sen)))
                print("true labels: ", util.get_lab_name(gold[i], labels))
                print("predicted: ", util.get_lab_name(pre[i], labels))

    try:
        pre = tp * 1.0 / (tp + fp)
        rec = tp * 1.0 / (tp + fn)
        f = 2.0 * pre * rec / (pre + rec)
        print(pre, rec, f)
    except ZeroDivisionError:
        print("DIV BY 0 ", tp, fp, fn)
    
    return (pre, rec, f)



def eval_pico_word_bs(gold, res, n = 100, l = 0, r = 950, seed = 33):
    """
    use bootstrap re-sample
    """
    rs = np.random.RandomState()
    rs.seed(seed)
    a = rs.permutation(len(gold))
    b = [] # list of indices to use

    for index in a[l:r]:
        b.append(index)

    list_p = []; list_r = []; list_f = []

    for bs in range(n):
        c = sklearn.utils.resample(b, random_state = bs)
        cm = np.zeros( (2,2) )

        for index in c:
            g = gold[index]
            i = g[0]
            t = g[1]
            for x, y in zip(t, res[i]):
                cm[x][y] += 1
        p = cm[1][1] * 1.0 / (cm[1][1] + cm[0][1])
        r = cm[1][1] * 1.0 / (cm[1][1] + cm[1][0])
        f = 2 * p * r / (p + r)
        list_p.append(p); list_r.append(r); list_f.append(f)

    print(np.mean(list_p), np.mean(list_r), np.mean(list_f)) 
    print(np.std(list_p),  np.std(list_r),  np.std(list_f))
    return list_f



def eval_pico_word(gold, res, l = 0, r = 950, seed = 33):
    rs = np.random.RandomState()
    rs.seed(seed)
    a = rs.permutation(len(gold))

    cm = np.zeros( (2,2) )
    for index in a[l:r]:
        g = gold[index]
        i = g[0]
        t = g[1]
        for x, y in zip(t, res[i]):
            cm[x][y] += 1
    p = cm[1][1] * 1.0 / (cm[1][1] + cm[0][1])
    r = cm[1][1] * 1.0 / (cm[1][1] + cm[1][0])
    f = 2 * p * r / (p + r)
    print(p, r, f)
    return cm

def eval_pico_entity(gold, res, l = 0, r = 950, seed = 33):
    tp = 0; fn = 0; fp = 0
    
    rs = np.random.RandomState()
    rs.seed(seed)
    a = rs.permutation(len(gold))

    for index in a[l:r]:
        g = gold[index]
        i = g[0]
        t = g[1]
        ge = list_entities(t, 1, 1)
        re = list_entities(res[i], 1, 1)
        
        for loc in ge:
            if loc in re:
                tp += 1
            else:
                fn += 1

        for loc in re:
            if loc not in ge:
                fp += 1

    pre = tp * 1.0 / (tp + fp)
    rec = tp * 1.0 / (tp + fn)
    f = 2.0 * pre * rec / (pre + rec)
    print(pre, rec, f)


def longest_x(a, x=0):
    i = 0
    n = len(a)
    res = 0
    while (i < n):
        while (i < n and a[i] != x): i += 1
        start = i
        while (i < n and a[i] == x): i += 1
        if i - start > res: res = i - start
    return res

def eval_pico_new(gold, res, l=0, r=950, dic_res = False):
    """
    eval PICO using three metrics.

    ***This seems to match with their results reported for the PICO dataset.***

    :param gold:
    :param res:
    :param l:
    :param r:
    :param dic_res: is res a dictionary
    :return:
    """
    tp = 0;
    fn = 0;
    fp = 0

    list_rec = []
    list_sr = []
    list_spe = []
    list_pre = []

    for index in range(l, r, 1):
        g = gold[index]
        i = g[0]
        if dic_res and i not in res: continue
        (t1, t2, t3) = eval_pico_sen(g[1], res[i])
        list_pre.extend(t1)
        list_rec.extend(t2)
        list_spe.extend(t3)

    if len(list_pre) == 0: list_pre.append(0)
    res = eval_pico_summary(list_pre, list_rec, list_spe)
    print(res)
    return res[3]

def eval_pico_sen(gold, res):
    """
    eval a PICO sentence
    :param gold:
    :param res:
    :return:
    """
    ge = list_entities(gold, 1, 1)  # gold entities
    neg = list_entities(gold, 0, 0)  # negative entities
    re = list_entities(res, 1, 1)  # entities in the predicted

    list_rec = []; list_pre = []; list_spe = []

    for (x, y) in ge:
        # coverage, loss of coverage:
        l = y - x + 1
        rec = np.sum(res[x:y + 1]) * 1.0 / l
        # loss = longeset_zeros(res[i][x:y+1])
        #sr = longest_x(res[x:y + 1], 1) * 1.0 / l
        list_rec.append(rec)
        #list_sr.append(sr)

    # precision
    for (x, y) in re:
        l = y - x + 1
        pre = np.sum(gold[x:y + 1]) * 1.0 / l
        list_pre.append(pre)
    # specificity:
    for (x, y) in neg:
        l = y - x + 1
        spe = (l - np.sum(res[x:y + 1])) * 1.0 / l
        list_spe.append(spe)

    return (list_pre, list_rec, list_spe)

def eval_pico_summary(list_pre, list_rec, list_spe, n_bs = 100):
    if len(list_pre) == 0: list_pre.append(0)
    lp = []; lr = []; ls = []; lf = []

    for bs in range(n_bs):
        bs_pre = sklearn.utils.resample(list_pre, random_state=bs)
        bs_rec = sklearn.utils.resample(list_rec, random_state=bs)
        bs_spe = sklearn.utils.resample(list_spe, random_state=bs)
        p = np.mean(bs_pre); lp.append(p)
        r = np.mean(bs_rec); lr.append(r)
        s = np.mean(bs_spe); ls.append(s)
        f = 2.0 * p * r / (p + r); lf.append(f)

    return ( np.mean(lp), np.mean(lr), np.mean(ls), np.mean(lf), \
             np.std(lp), np.std(lr), np.std(ls), np.std(lf) )

def eval_pico_test(gold, f, sentences, features, l = 0, r = 950):
    """

    :param gold:
    :param f: prediction model
    :return:
    """
    inv_f = {v: k for (k, v) in list(features.items())}
    list_pre = []
    list_rec = []
    list_spe = []

    #res = [ [] for i in range(len(sentences))]
    for i, g in gold[l:r]:
        sen = sentences[i]
        for ins in sen:
            ins.word = inv_f[ins.features[0]]
        predicted = get_tag(f, sen, None, 't')
        #print predicted, g
        #res[i] = predicted
        (t1, t2, t3) = eval_pico_sen(g, predicted)
        list_pre.extend(t1)
        list_rec.extend(t2)
        list_spe.extend(t3)

    if len(list_pre) == 0: list_pre.append(0)
    res = eval_pico_summary(list_pre, list_rec, list_spe)
    print(res)
    return res[3]


def main():
    hc, all_sen, features, labels = run_rod()
    hc.learn(0)
    eval_hc_test(hc, features, labels, print_err=True)


def main2():
    hc, all_sen, features, labels = run_rod()
    h = HMM(len(labels) + 1, len(features) + 1)
    mv_sen, cor, wrong = util.mv_cd(hc.data, labels)
    h.learn(mv_sen)
    h.decode_all(mv_sen)
    eval_seq_train(util.get_all_lab(all_sen), h.res, labels)


def output_to_file(data, res, features, labels, output_file):
    f = open(output_file, 'w')
    inv_l = {v: k for k, v in list(labels.items())}
    inv_f = {v: k for k, v in list(features.items())}
    
    for sen, lab in zip(data.sentences, res):
        wl = util.get_word_list2(sen, inv_f)
        ll = util.get_lab_name_list2(lab, inv_l)
        for w, l in zip(wl, ll):
            f.write(w + ' ' + l + '\n')
        f.write('\n')
    
    f.close()

def main_em(opts):
    """
    run em, eval, print
    """
    hs, hc, all_sen, features, labels = run_rod(dirname=opts.inputdir)
    hc.init(init_type = 'dw', wm_rep='cv2', dw_em=5, wm_smooth=float(opts.smooth))
    gold = util.get_all_lab(all_sen)

    hc.vb = [float(opts.variational_prior), float(opts.variational_prior)]
    hc.em(int(opts.emsteps))
    hc.mls()
    
    output_to_file(hc.data, hc.res, features, labels, opts.output)
    
    print(eval_seq_train(gold, hc.res, labels))


import sys
import optparse

if __name__ == "__main__":

    optparser = optparse.OptionParser()
    optparser.add_option(
    "-i", "--inputdir", default="task1/val/",
    help="input directory"
    )

    optparser.add_option(
    "-o", "--output", default="output.txt",
    help="output file"
    )
    
    optparser.add_option(
    "-e", "--emsteps", default="1",
    help="number of EM steps"
    )
    
    optparser.add_option(
    "-s", "--smooth", default="0.001",
    help="smooth parameters"
    )
    
    optparser.add_option(
    "-v", "--variational_prior", default="0.1",
    help="prior for the variational estimator"
    )

    opts = optparser.parse_args()[0]
    
    main_em(opts)
