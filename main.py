import graphNLP as gnlp


texts, labels_ = gnlp.preprocessing(path_neg='\\dataset\\neg', path_pos='\\dataset\\pos')

train_doc, labels_doc = gnlp.preprocessing(path_neg='\\train_doc2vec\\neg', path_pos='\\train_doc2vec\\pos')
dv_model = gnlp.docs2vecs(train_doc)
gnlp.save_to_file(dv_model)
# or
# dv_model = Doc2Vec.load('my_doc2vec_model')

A = gnlp.build_graph(texts, gnlp.build_edges(dv_model, texts, 0.80))

X, N, F = gnlp.build_feature_matrix(texts)

labels_encoded, classes = gnlp.encode_label(labels_)

train_mask, val_mask, test_mask = gnlp.get_masks(N, labels_)

dataset = gnlp.create_dataset(X, A, labels_encoded)

weights_tr, weights_va, weights_te = (
    gnlp.mask_to_weights(mask)
    for mask in (train_mask, val_mask, test_mask)
)

model = gnlp.build_and_train_GCN_model(dataset, N, weights_tr, weights_va)

gnlp.evaluating_model(model, dataset, weights_te)
