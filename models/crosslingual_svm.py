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

from scipy.stats import pearsonr

def add_unknown_words(wordvecs, vocab, min_fq=10, dim=50):
    """
    For words that occur less than min_fq, create a separate word vector
    0.25 is chosen so the unk vectors have approximately the same variance
    as pretrained ones
    """
    for word in vocab:
        if word not in wordvecs and vocab[word] >= min_fq:
            wordvecs[word] = np.random.uniform(-0.25, 0.25, dim)


def get_W(wordvecs, dim=300):
    """
    Returns a word matrix W where W[i] is the vector for word indexed by i
    and a word-to-index dictionary w2idx, whose keys are words and whose
    values are the indices.
    """
    vocab_size = len(wordvecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, dim), dtype='float32')

    # set unk to 0
    word_idx_map['UNK'] = 0
    W[0] = np.zeros(dim, dtype='float32')
    i = 1
    for word in wordvecs:
        W[i] = wordvecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def ave_sent(sent, w2idx, matrix):
    """
    Converts a sentence to the mean
    embedding of the word vectors found
    in the sentence.
    """
    array = []
    for w in sent:
        try:
            array.append(matrix[w2idx[w]])
        except KeyError:
            array.append(np.zeros(300))
    return np.array(array).mean(0)

def convert_svm_dataset(dataset, w2idx, matrix):
    """
    Change dataset representation from a list of lists, where each outer list
    is a sentence and each inner list contains the tokens. The result
    is a matrix of size n x m, where n is the number of sentences
    and m = the dimensionality of the embeddings in the embedding matrix.
    """
    try:
        dataset._Xtrain = np.array([ave_sent(s, w2idx, matrix) for s in dataset._Xtrain])
    except:
        print("No train set")
        pass
    try:
        dataset._Xdev = np.array([ave_sent(s, w2idx, matrix) for s in dataset._Xdev])
    except:
        print("No dev set")
        pass
    try:
        dataset._Xtest = np.array([ave_sent(s, w2idx, matrix) for s in dataset._Xtest])
    except:
        print("No test set")
        pass
    return dataset

def get_projection_matrix(pdataset, src_vecs, trg_vecs):
    X, Y = [], []
    for i in pdataset._Xtrain:
        X.append(src_vecs[i])
    for i in pdataset._ytrain:
        Y.append(trg_vecs[i])

    X = np.array(X)
    Y = np.array(Y)
    u, s, vt = np.linalg.svd(np.dot(Y.T, X))
    W = np.dot(vt.T, u.T)
    return W

def str2bool(v):
    # Converts a string to a boolean, for parsing command line arguments
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', default='es', help='choose target language: es, ca, eu (defaults to es)')
    parser.add_argument('-se', '--src_embedding', default="../../embeddings/blse/google.txt")
    parser.add_argument('-te', '--trg_embedding', default="../../embeddings/blse/sg-300-es.txt")
    parser.add_argument('-sd', '--src_dataset', default="../dataset/en")
    parser.add_argument('-td', '--trg_dataset', default="../dataset/es/original")
    parser.add_argument('-emo', '--emotion', default="anger")

    args = parser.parse_args()

    # Import monolingual vectors
    print('importing word embeddings')
    src_vecs = WordVecs(args.src_embedding)
    src_vecs.mean_center()
    src_vecs.normalize()
    trg_vecs = WordVecs(args.trg_embedding)
    trg_vecs.mean_center()
    trg_vecs.normalize()

    # Setup projection dataset
    trans = '../lexicons/bingliu/en-{0}.txt'.format(args.lang)
    pdataset = ProjectionDataset(trans, src_vecs, trg_vecs)

    # learn the translation matrix W
    print('Projecting src embeddings to trg space...')
    W = get_projection_matrix(pdataset, src_vecs, trg_vecs)
    print('W done')

    # project the source matrix to the new shared space
    src_vecs._matrix = np.dot(src_vecs._matrix, W)
    print('src_vecs done')

    # open datasets
    src_dataset = Emotion_Dataset(args.src_dataset, None, rep=words, emotion=args.emotion)
    print('src_dataset done')
    trg_dataset = Emotion_Testset(args.trg_dataset, None, rep=words, emotion=args.emotion)
    print('trg_dataset done')

    # get joint vocabulary and maximum sentence length
    print('Getting joint space and vocabulary...')
    max_length = 0
    src_vocab = {}
    trg_vocab = {}
    for sentence in list(src_dataset._Xtrain) + list(src_dataset._Xdev):
        if len(sentence) > max_length:
            max_length = len(sentence)
        for word in sentence:
            if word in src_vocab:
                src_vocab[word] += 1
            else:
                src_vocab[word] = 1
    for sentence in list(trg_dataset._Xtest):
        if len(sentence) > max_length:
            max_length = len(sentence)
        for word in sentence:
            if word in trg_vocab:
                trg_vocab[word] += 1
            else:
                trg_vocab[word] = 1


    # get joint embedding space
    joint_embeddings = {}
    for vecs in [src_vecs, trg_vecs]:
        for w in vecs._w2idx.keys():
            # if a word is found in both source and target corpora,
            # choose the version with the highest frequency
            if w in src_vocab and w in src_vecs and w in trg_vocab and w in trg_vecs:
                if src_vocab[w] >= trg_vocab[w]:
                    joint_embeddings[w] = src_vecs[w]
                else:
                    joint_embeddings[w] = trg_vecs[w]
            elif w in src_vocab and w in src_vecs:
                joint_embeddings[w] = src_vecs[w]
            elif w in trg_vocab and w in trg_vecs:
                joint_embeddings[w] = trg_vecs[w]

    joint_vocab = {}
    joint_vocab.update(src_vocab)
    joint_vocab.update(trg_vocab)

    add_unknown_words(joint_embeddings, joint_vocab, min_fq=1, dim=300)
    joint_matrix, joint_w2idx = get_W(joint_embeddings, dim=300)

    # save the w2idx and max length
    paramdir = os.path.join("saved_models",
                             args.lang,
                             args.emotion)

    os.makedirs(paramdir, exist_ok=True)

    with open(os.path.join(paramdir, "w2idx.pkl"), 'wb') as out:
        pickle.dump((joint_w2idx, joint_matrix), out)
    print('Saved joint vocabulary...')

    # convert datasets
    src_dataset = convert_svm_dataset(src_dataset, joint_w2idx, joint_matrix)
    trg_dataset = convert_svm_dataset(trg_dataset, joint_w2idx, joint_matrix)

    # train Support Vector Regression on source
    print('Training SVR...')
    checkpoint = os.path.join("saved_models", args.lang,
                                  args.emotion,
                                  "vecmap",
                                  "svr")

    os.makedirs(checkpoint, exist_ok=True)
    clf = SVR(C=100, kernel="linear")
    history = clf.fit(src_dataset._Xtrain, src_dataset._ytrain)
    with open(os.path.join(checkpoint, "weights.pkl"), 'wb') as out:
        pickle.dump(clf, out)

    # test on src devset and trg devset
    src_pred = clf.predict(src_dataset._Xdev)
    score, p = pearsonr(src_dataset._ydev, src_pred)
    print("SRC-SRC: {0:.3f} ({1:.3f})".format(score, p))

    trg_pred = clf.predict(trg_dataset._Xtest)
    score, p = pearsonr(trg_dataset._ytest, trg_pred)
    print("SRC-TRG: {0:.3f} ({1:.3f})".format(score, p))

