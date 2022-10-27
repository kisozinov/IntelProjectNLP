import graphNLP as gnlp
from gensim.models.doc2vec import Doc2Vec
import time
import numpy as np
from keras.models import load_model

# texts, labels_ = gnlp.preprocessing(path_neg='\\dataset\\neg', path_pos='\\dataset\\pos')
# np.save('labels.npy', labels_)
labels_ = np.load('labels.npy')
# train_doc, labels_doc = gnlp.preprocessing(path_neg='\\train_doc2vec\\neg', path_pos='\\train_doc2vec\\pos')
# dv_model = gnlp.docs2vecs(train_doc)
# gnlp.save_to_file(dv_model)
# or
dv_model = Doc2Vec.load('my_doc2vec.model')

#A = gnlp.build_graph(texts, gnlp.build_edges(dv_model, texts, 0.55))
#np.save('adj_matrix', A)
A = np.load('adj_matrix.npy')
#print(A.shape[0], A.shape[1])
#X, N, F = gnlp.build_feature_matrix(texts)
#np.save('feature_matrix', X)
X = np.load('feature_matrix.npy')
N = X.shape[0]  # the number of nodes
F = X.shape[1]  # the size of node features
print(X)

labels_encoded, classes = gnlp.encode_label(labels_)

train_mask, val_mask, test_mask = gnlp.get_masks(N, labels_)
print(test_mask)
# dataset = gnlp.create_dataset(X, A, labels_encoded) #X, A, labels_encoded
dataset = gnlp.MyDataset(X, A, labels_encoded)

weights_tr, weights_va, weights_te = (
    gnlp.mask_to_weights(mask)
    for mask in (train_mask, val_mask, test_mask)
)
print(weights_te)
start_time = time.time()
model = gnlp.build_and_train_GCN_model(dataset, N, weights_tr, weights_va)

gnlp.evaluating_model(model, dataset, weights_te)
print('Time: ', time.time() - start_time)


model.save('gcn_model.tf', save_format='tf')
