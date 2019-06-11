import os, re
import numpy as np

# =========================================================
# Representations
# =========================================================

class ProjectionDataset():
    """
    A wrapper for the translation dictionary. The translation dictionary
    should be word to word translations separated by a tab. The
    projection dataset only includes the translations that are found
    in both the source and target vectors.
    """
    def __init__(self, translation_dictionary, src_vecs, trg_vecs):
        (self._Xtrain, self._Xdev, self._ytrain,
         self._ydev) = self.getdata(translation_dictionary, src_vecs, trg_vecs)

    def getdata(self, translation_dictionary, src_vecs, trg_vecs):
        x, y = [], []
        with open(translation_dictionary) as f:
            for line in f:
                src, trg = line.split()
                try:
                    _ = src_vecs[src]
                    _ = trg_vecs[trg]
                    x.append(src)
                    y.append(trg)
                except:
                    pass
        xtr, xdev = self.train_dev_split(x)
        ytr, ydev = self.train_dev_split(y)
        return xtr, xdev, ytr, ydev

    def train_dev_split(self, x, train=.9):
        # split data into training and development, keeping /train/ amount for training.
        train_idx = int(len(x)*train)
        return x[:train_idx], x[train_idx:]

def ave_vecs(sentence, model):
    sent = np.array(np.zeros((model.vector_size)))
    sent_length = len(sentence.split())
    for w in sentence.split():
        try:
            sent += model[w]
        except:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent += model['the']
    return sent / sent_length


def idx_vecs(sentence, model, lowercase=True):
    """Returns a list of vectors of the tokens
    in the sentence if they are in the model."""
    sent = []
    if lowercase:
        sentence = sentence.lower()
    for w in sentence.split():
        try:
            sent.append(model[w])
        except KeyError:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent.append(model['UNK'])
    return sent


def words(sentence, model):
    return sentence.split()


def raw(sentence, model):
    return sentence

def bow(sentence, model):
    a = np.zeros(len(model))
    for i in sentence.split():
        try:
            idx = model._w2idx[i]
            a[idx] += 1
        except:
            a[0] += 1
    return a

# =========================================================
# Open a single data file
# =========================================================


def getTrainData(fname, model, representation=ave_vecs, encoding='utf8'):
    data = []
    for sent in open(fname):
        idx, text, emotion, label = sent.strip().split("\t")
        data.append((representation(text, model), float(label)))
    return data

def getTestData(fname, model, representation=ave_vecs, encoding='utf8'):
    data = []
    for i, sent in enumerate(open(fname)):
        try:
            text, label = sent.strip().split("\t")
            data.append((representation(text, model), float(label)))
        except:
            print(i)
            print(text)
    return data


class Emotion_Dataset(object):

    def __init__(self, DIR, model,
                 dtype=np.float32,
                 rep=None,
                 emotion="anger"):

        self.rep = rep

        Xtrain, Xdev, ytrain, ydev = self.open_data(DIR, model, rep, emotion)


        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._num_examples = len(self._Xtrain)

    def open_data(self, DIR, model, rep, emotion):
        train = getTrainData(os.path.join(DIR, "train", "{0}.txt".format(emotion)),
                          model, rep)
        dev = getTrainData(os.path.join(DIR, "dev", "{0}.txt".format(emotion)),
                          model, rep)

        Xtrain, ytrain = zip(*train)
        Xdev, ydev = zip(*dev)

        return Xtrain, Xdev, ytrain, ydev

class Emotion_Testset(object):

    def __init__(self, DIR, model,
                 dtype=np.float32,
                 rep=None,
                 emotion="anger"):

        self.rep = rep

        Xtest, ytest= self.open_data(DIR, model, rep, emotion)


        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtest)

    def open_data(self, DIR, model, rep, emotion):
        test = getTestData(os.path.join(DIR, "test", "{0}.txt".format(emotion)),
                          model, rep)

        Xtest, ytest = zip(*test)

        return Xtest, ytest

