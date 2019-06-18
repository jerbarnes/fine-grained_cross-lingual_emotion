from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

from Utils.WordVecs import *
from Utils.Datasets import *

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import numpy as np
import sys
import argparse
import pickle
import json
import re

from collections import defaultdict
from scipy.stats import pearsonr

def bow(x, vocab):
    a = np.zeros((len(vocab)))
    for i in x:
        a[i] += 1
    return a


class Vocab(defaultdict):
    def __init__(self, train=True):
        super(Vocab, self).__init__(lambda : len(self))
        self.train = train
        self.UNK = "UNK"
        # set UNK token to 0 index
        self[self.UNK]

    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return [self[w] for w in ws]
        else:
            return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if i in idx2w else "UNK" for i in ids]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', default='ca', help='choose target language: es, ca, eu (defaults to ca)')
    parser.add_argument('-sd', '--src_dataset', default="../dataset/en")
    parser.add_argument('-td', '--trg_dataset', default="../dataset/ca/trans")
    parser.add_argument('-emo', '--emotion', default="anger")

    args = parser.parse_args()


    # open datasets
    src_dataset = Emotion_Dataset(args.src_dataset, None, rep=words, emotion=args.emotion)
    print('src_dataset done')
    trg_dataset = Emotion_Testset(args.trg_dataset, None, rep=words, emotion=args.emotion)
    print('trg_dataset done')

    # Start Vocab
    vocab = Vocab()

    # convert datasets
    src_dataset._Xtrain = [vocab.ws2ids(s) for s in src_dataset._Xtrain]
    src_dataset._Xdev = [vocab.ws2ids(s) for s in src_dataset._Xdev]
    trg_dataset._Xtest = [vocab.ws2ids(s) for s in trg_dataset._Xtest]

    src_dataset._Xtrain = [bow(s, vocab) for s in src_dataset._Xtrain]
    src_dataset._Xdev = [bow(s, vocab) for s in src_dataset._Xdev]
    trg_dataset._Xtest = [bow(s, vocab) for s in trg_dataset._Xtest]

    # train Support Vector Regression on source
    print('Training SVR...')
    clf = SVR(C=100, kernel="linear")
    history = clf.fit(src_dataset._Xtrain, src_dataset._ytrain)

    # test on src devset and trg devset
    src_pred = clf.predict(src_dataset._Xdev)
    score, p = pearsonr(src_dataset._ydev, src_pred)
    print("SRC-SRC: {0:.3f} ({1:.3f})".format(score, p))

    trg_pred = clf.predict(trg_dataset._Xtest)
    score, p = pearsonr(trg_dataset._ytest, trg_pred)
    print("SRC-TRG: {0:.3f} ({1:.3f})".format(score, p))

