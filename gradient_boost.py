import os
from gensim.models import Doc2Vec
import graphNLP as gnlp
import daal4py as d4p
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time




dv_model = Doc2Vec.load('my_doc2vec.model')
print(dv_model.dv)

texts, labels_ = gnlp.preprocessing(path_neg='\\dataset\\neg', path_pos='\\dataset\\pos')

# node_vectors = []
# for text in texts:
#     node_vectors.append(dv_model.infer_vector(text))
#
#

node_vectors = np.load('vectors.npy')
data = []
for i in range(len(node_vectors)):
    data.append(np.append(node_vectors[i], labels_[i]))

train_data, test_data = train_test_split(data, test_size=0.3)

np.savetxt('train_data.txt', train_data)
np.savetxt('test_data.txt', test_data)



def read_csv(f, c=None, t=np.float64):
    return np.loadtxt(f, usecols=c, delimiter=' ', ndmin=2, dtype=t)


def main(readcsv=read_csv, method='defaultDense'):
    nFeatures = 15
    nClasses = 2
    maxIterations = 200
    minObservationsInLeafNode = 8
    # input data file
    infile = "train_data.txt"
    testfile = "test_data.txt"

    # Configure a training object (5 classes)
    train_algo = d4p.gbt_classification_training(
        nClasses=nClasses,
        maxIterations=maxIterations,
        minObservationsInLeafNode=minObservationsInLeafNode,
        featuresPerNode=nFeatures,
        varImportance='weight|totalCover|cover|totalGain|gain'
    )
    # train_algo = d4p.kdtree_knn_classification_training(nClasses=nClasses)
    # Read data. Let's use 3 features per observation
    data = readcsv(infile, range(15), t=np.float32)
    labels = readcsv(infile, range(15, 16), t=np.float32)
    #weights = np.ones((data.shape[0], 1))
    train_result = train_algo.compute(data, labels)

    # Now let's do some prediction
    # previous version has different interface
    predict_algo = d4p.gbt_classification_prediction(
        nClasses=nClasses,
    )
    # read test data (with same #features)
    pdata = readcsv(testfile, range(15), t=np.float32)
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    plabels = readcsv(testfile, range(15, 16), t=np.float32)
    #assert np.count_nonzero(predict_result.prediction - plabels) / pdata.shape[0] < 0.022

    return (train_result, predict_result, plabels)


# if __name__ == "__main__":
#     start = time.time()
#     (train_result, predict_result, plabels) = main()
#     print(
#         "\nGradient boosted trees prediction results (first 10 rows):\n",
#         predict_result.prediction[0:10]
#     )
#     print("\nGround truth (first 10 rows):\n", plabels[0:10])
#
#
#     print(accuracy_score(plabels, predict_result.prediction))
#     print(precision_score(plabels, predict_result.prediction))
#     print(recall_score(plabels, predict_result.prediction))
#     print(time.time() - start)
