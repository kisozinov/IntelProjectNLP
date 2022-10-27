import nltk
# use it at the first launch
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import graphviz
import networkx as nx
import json
import os
import random

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim

import torch as th
# import dgl
# from dgl.nn import GraphConv

from spektral.layers import GCNConv, GlobalSumPool
from spektral.models.gcn import GCN
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense

from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from spektral.data import Dataset
from spektral.data import Graph
from spektral.data import SingleLoader
from spektral.data import BatchLoader
from random import shuffle
from scipy import spatial
from gensim.test.utils import get_tmpfile


def preprocessing(path_neg='\\dataset\\neg', path_pos='\\dataset\\pos'):
    print('Current working directory:', os.getcwd())
    texts = []
    labels_ = []
    nodes_count = 0
    punct = '".#$%&\'*+,-/:;<=>@[\\]^_`{|}~!'
    seed = 111

    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    lemmatizer = WordNetLemmatizer()

    for filename in os.listdir(str(os.getcwd() + path_neg)):
        with open(str(os.getcwd() + path_neg + '\\' + filename), 'r', encoding='utf-8') as f:
            nodes_count += 1
            raw_text = f.read()
            for p in punct:
                if p in raw_text:
                    raw_text = raw_text.replace(p, '')
                    raw_text = raw_text.replace(',', '')
            lower_text = raw_text.lower()
            my_stopwords = [w.casefold() for w in stopwords.words()]
            texts_lem = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(lower_text) if
                         not w.casefold() in my_stopwords]
            texts.append(texts_lem)
            labels_.append(0)

    for filename in os.listdir(str(os.getcwd() + path_pos)):
        with open(str(os.getcwd() + path_pos + '\\' + filename), 'r', encoding='utf-8') as f:
            nodes_count += 1
            raw_text = f.read()
            for p in punct:
                if p in raw_text:
                    raw_text = raw_text.replace(p, '')
                    raw_text = raw_text.replace(',', '')
            lower_text = raw_text.lower()
            my_stopwords = [w.casefold() for w in stopwords.words()]
            texts_lem = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(lower_text) if
                         not w.casefold() in my_stopwords]
            texts.append(texts_lem)
            labels_.append(1)

    labels_ = np.array(np.array(labels_, dtype=float))
    labels_ = np.transpose(labels_)
    print('Preprocessing is done.\n')
    if len(texts) == len(labels_):
        random.Random(seed).shuffle(texts)
        random.Random(seed).shuffle(labels_)
    else:
        return 0
    return texts, labels_


def single_preprocessing(raw_text):
    punct = '".#$%&\'*+,-/:;<=>@[\\]^_`{|}~!'

    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    lemmatizer = WordNetLemmatizer()

    for p in punct:
        if p in raw_text:
            raw_text = raw_text.replace(p, '')
            raw_text = raw_text.replace(',', '')
    lower_text = raw_text.lower()
    my_stopwords = [w.casefold() for w in stopwords.words()]
    texts_lem = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(lower_text) if
                 not w.casefold() in my_stopwords]

    return texts_lem

def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


def docs2vecs(texts):
    # N = len(list(tagged_document(texts)))
    train_data = list(tagged_document(texts))  # [:int(N/2)]
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=30)
    print('Doc2vec model is successfulely built.\n')
    model.build_vocab(train_data)
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
    print('Doc2vec model is successfulely trained.\n')
    return model


def save_to_file(model):
    model.save("my_doc2vec.model")


def my_most_similar(vectors, tag, min_val):
    result = []
    for i in range(len(vectors)):
        if tag != i:
            cos_sim = 1 - spatial.distance.cosine(vectors[tag], vectors[i])
            if cos_sim >= min_val:
                result.append((i, cos_sim))
    return result


def build_edges(dv_model, texts, threshsold):
    all_edges = set()
    node_vectors = []
    for text in texts:
        node_vectors.append(dv_model.infer_vector(text))
    np.save('vectors.npy', node_vectors)
    for tag in range(len(texts)):
        for key, value in my_most_similar(node_vectors, tag, threshsold):  ### 0.86 на 15
            all_edges.add((tag, key))
    return all_edges


def build_graph(texts, edges):
    nodes = list(range(len(texts)))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    print('Graph is successfuly built.')
    A = nx.to_numpy_array(G, dtype=float)  # A = nx.adjacency_matrix(G)
    print('Graph info: ', nx.info(G), '\n')
    return A


def build_feature_matrix(texts):
    words_set = set()
    for text in texts:
        words_set = words_set.union(set(text))
    np.save('words_set.npy', np.array(list(words_set)))
    X = []
    for word in words_set:
        col = []
        for text in texts:
            if word in text:
                col.append(1)
            else:
                col.append(0)
        X.append(col)
        col = []
    X = np.array(X, dtype=float).transpose()

    N = X.shape[0]  # the number of nodes
    F = X.shape[1]  # the size of node features
    print('Feature matrix is successfully built')
    return X, N, F


def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    print('Labels converted to one-hot encoding')
    return labels, label_encoder.classes_


def get_masks(N, labels_):
    def limit_data(labels, limit=1500, val_num=1000, test_num=1000):
        '''
        Get the index of train, validation, and test data
        '''
        label_counter = dict((l, 0) for l in labels)
        train_idx = []

        for i in range(len(labels)):
            label = labels[i]
            if label_counter[label] < limit:
                # add the example to the training data
                train_idx.append(i)
                label_counter[label] += 1

            # exit the loop once we found limit examples for each class
            if all(count == limit for count in label_counter.values()):
                break

        # get the indices that do not go to traning data
        rest_idx = [x for x in range(len(labels)) if x not in train_idx]
        val_idx = rest_idx[:val_num]
        test_idx = rest_idx[val_num:(val_num + test_num)]
        return train_idx, val_idx, test_idx

    '''
    total = len(labels_)
    limit = int(total * 0.6 / 2)
    val_num = int(total * 0.2)
    test_num = int(total * 0.2)
    print(limit*2, val_num, test_num)
    '''
    # train_idx,val_idx,test_idx = limit_data(labels_, limit, val_num, test_num)
    train_idx, val_idx, test_idx = limit_data(labels_)

    # set the mask
    train_mask = np.zeros((N,), dtype=bool)
    train_mask[train_idx] = True

    val_mask = np.zeros((N,), dtype=bool)
    val_mask[val_idx] = True

    test_mask = np.zeros((N,), dtype=bool)
    test_mask[test_idx] = True

    print('Masks for data received.')
    return train_mask, val_mask, test_mask


def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)


class MyDataset(Dataset):

    def __init__(self, feats=[[]], adj_matrix=[[]], labels=[[]], **kwargs):
        self.adj = adj_matrix
        self.x = feats
        self.labels = labels
        #self.nodes = nodes
        #self.feats = feats
        #self.labels = labels
        super().__init__(**kwargs)

    def download(self):
        # data = ...  # Download from somewhere

        # Create the directory
        os.mkdir(self.path)
        # Write the data to file
        filename = os.path.join(self.path, f'graph')
        x = self.x
        a = self.adj
        y = self.labels
        np.savez(filename, x=x, a=a, y=y)

    def read(self):
        # We must return a list of Graph objects
        output = []
        print(self.path)
        data = np.load(os.path.join(self.path, 'graph.npz'), allow_pickle=True)
        print(data['x'], data['a'], data['y'])
        output.append(
            Graph(x=data['x'], a=data['a'], y=data['y'])
        )

        return output

    def get_info(self):
        return self.graphs

    def add(self, inp_a, inp_x):
        self.graphs[0].a = np.append(self.graphs[0].a, inp_a, axis=0)
        inp_a_col = np.append(inp_a[0], 1).reshape(len(inp_a[0]) + 1, 1)
        self.graphs[0].a = np.append(self.graphs[0].a, inp_a_col, axis=1)
        self.graphs[0].x = np.append(self.graphs[0].x, inp_x, axis=0)


def create_dataset(X, A, labels_encoded):
    return MyDataset(X, A, labels_encoded)


def build_and_train_GCN_model(dataset, N, weights_tr, weights_va):
    loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
    loader_va = SingleLoader(dataset, sample_weights=weights_va)
    print('Loaders is ready')

    learning_rate = 1e-3
    epochs = 40
    patience = 20

    model = GCN(n_labels=dataset.n_labels, n_input_channels=dataset.n_node_features)
    model.compile(
        optimizer=Adam(learning_rate),
        loss=BinaryCrossentropy(reduction="sum"),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()], # tf.keras.metrics.Precision()
        weighted_metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.fit(
        loader_tr.load(),
        batch_size=N,
        steps_per_epoch=loader_tr.steps_per_epoch,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
    )
    return model


def evaluating_model(model, dataset, weights_te):
    # Evaluate model
    print("Evaluating model.")
    loader_te = SingleLoader(dataset, sample_weights=weights_te)
    eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    print("Done.\n" "Test loss: {}\n" "Test accuracy: {} " "Test Precision: {} " "Test Recall: {} " "Test ROC AUC: {} ".format(*eval_results))
    print(model.predict(loader_te.load(), steps=loader_te.steps_per_epoch))
