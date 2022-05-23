import nltk
# use it at the first launch
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import os
import numpy as np
import random

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim


def preprocessing(filename):
    print('Current working directory:', os.getcwd())
    texts = []
    labels_ = []
    nodes_count = 0
    punct = '".#$%&\'*+,-/:;<=>@[\\]^_`{|}~!?'
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

    with open(str(os.getcwd() + '\\amazon_dataset' + '\\' + filename), 'r', encoding='utf-8') as f:
        for i in range(5000):
            nodes_count += 1
            label = f.readline(10)
            raw_text = f.readline()

            for p in punct:
                if p in raw_text:
                    raw_text = raw_text.replace(p, '')
                    raw_text = raw_text.replace(',', '')
            lower_text = raw_text.lower()
            my_stopwords = [w.casefold() for w in stopwords.words()]
            texts_lem = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(lower_text) if
                         not w.casefold() in my_stopwords]
            texts.append(texts_lem)
            if label[-1] == '1':
                labels_.append(0)
            else:
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


def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


def docs2vecs(texts):
    # N = len(list(tagged_document(texts)))
    train_data = list(tagged_document(texts))  # [:int(N/2)]
    model = gensim.models.doc2vec.Doc2Vec(vector_size=15, min_count=2, epochs=30)
    print('Doc2vec model is successfulely built.\n')
    model.build_vocab(train_data)
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
    print('Doc2vec model is successfulely trained.\n')
    return model


def save_to_file(model):
    model.save("amazon_doc2vec.model")


# texts_train, labels_ = preprocessing('train_5000.txt')
# print(labels_[:50])

# dv_model = docs2vecs(texts_train)
# save_to_file(dv_model)

dv_model = Doc2Vec.load('amazon_doc2vec.model')

texts, labels_ = preprocessing('work_5001-10000.txt')
print(labels_[:50])

inferred_vectors = []
for text in texts:
    inferred_vectors.append(dv_model.infer_vector(text))

print(inferred_vectors[:5])
