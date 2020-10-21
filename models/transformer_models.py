import torch
from torch import nn
from torch.nn import MSELoss
from transformers import BertModel, BertTokenizer, XLMRobertaTokenizer, XLMRobertaModel
from Utils.Datasets import *
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
import argparse
from sklearn.svm import SVR
import random


def dev_model(bert_model, classifier, dataset):
    dev_preds = []
    dev_x = [" ".join(l) for l in dataset._Xdev]
    dev_y = dataset._ydev

    i = 0
    num_batches = int(len(dev_x) / batch_size)
    if (len(dev_x) % batch_size) > 0:
        num_batches += 1
    dev_encs = []
    for idx in tqdm(range(num_batches), desc="Dev English"):
        x = dev_x[i:i+batch_size]
        i += batch_size
        encoding = tokenizer(x,
                             return_tensors='pt',
                             padding=True,
                             truncation=True)
        enc = bert_model(**encoding)[1]
        dev_encs.extend(enc.detach().numpy())
    #
    dev_preds = classifier.predict(dev_encs)
    score, p = pearsonr(src_dataset._ydev, dev_preds)
    print("SRC-SRC: {0:.2f} ({1:.2f})".format(score, p))
    return dev_preds

def test_model(bert_model, classifier, dataset, lang):
    test_dataset = Emotion_Testset(dataset,
                                   None,
                                   rep=words,
                                   emotion=args.emotion)
    preds = []
    test_x = [" ".join(l) for l in test_dataset._Xtest]
    test_y = test_dataset._ytest
    #
    i = 0
    num_batches = int(len(test_x) / batch_size)
    if (len(test_x) % batch_size) > 0:
        num_batches += 1
    test_encs = []
    for idx in tqdm(range(num_batches), desc="Test {0}".format(lang)):
        x = test_x[i:i + batch_size]
        i += batch_size
        encoding = tokenizer(x,
                             return_tensors='pt',
                             padding=True,
                             truncation=True)
        enc = bert_model(**encoding)[1]
        test_encs.extend(enc.detach().numpy())
    #
    preds = classifier.predict(test_encs)
    score, p = pearsonr(test_y, preds)
    print("SRC-{0}: {1:.2f} ({2:.2f})".format(lang, score, p))
    return preds



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--src_dataset', default="../dataset/en")
    parser.add_argument('-emo', '--emotion', default="anger")
    parser.add_argument('-e', '--epochs', default=5)
    parser.add_argument('--seed', default=1234)
    parser.add_argument('--pretrained_model', default="bert-base-multilingual-cased")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # open datasets
    src_dataset = Emotion_Dataset(args.src_dataset,
                                  None,
                                  rep=words,
                                  emotion=args.emotion)
    print('src_dataset done')

    if "roberta" in args.pretrained_model:
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
        model = XLMRobertaModel.from_pretrained(args.pretrained_model)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        model = BertModel.from_pretrained(args.pretrained_model)
    clf = SVR(C=100, kernel="linear")

    train_x = [" ".join(l) for l in src_dataset._Xtrain]
    train_y = src_dataset._ytrain


    batch_size = 20
    num_batches = int(len(train_x) / batch_size)
    if (len(train_x) % batch_size) > 0:
        num_batches += 1

    train_encs = []

    idxs = np.arange(len(train_x))
    np.random.shuffle(idxs)
    train_x = list(np.array(train_x)[idxs])
    train_y = list(np.array(train_y)[idxs])
    i = 0
    preds = []
    for idx in tqdm(range(num_batches)):
        x = train_x[i:i+batch_size]
        i += batch_size
        encoding = tokenizer(x,
                             return_tensors='pt',
                             padding=True,
                             truncation=True)
        enc = model(**encoding)[1]
        train_encs.extend(enc.detach().numpy())

    clf.fit(train_encs, train_y)
    preds = clf.predict(train_encs)
    score, p = pearsonr(train_y, preds)
    print("Score: {0:.2f} ({1:.2f})".format(score, p))

    # test on English
    en_preds = dev_model(model, clf, src_dataset)

    # test on Spanish
    es_preds = test_model(model, clf, "../dataset/es/original", "ES")

    # Test on Catalan
    ca_preds = test_model(model, clf, "../dataset/ca/original", "CA")
