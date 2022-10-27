import eel
import graphNLP as gnlp
from keras.models import load_model
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import accuracy_score, precision_score, recall_score
import daal4py as d4p
import random
import time

eel.init('web')

############################################
# model = load_model('gcn_model.tf')
# A = np.load('adj_matrix.npy')
# X = np.load('feature_matrix.npy')
# print('размероость матрицы х: ', X.shape[0], X.shape[1])
# dv_model = Doc2Vec.load('my_doc2vec.model')
# node_vectors = np.load('vectors.npy')
# words_set = np.load('words_set.npy')
#
# #inp = "I went to see this film out of curiosity, and to settle an argument. The film is now best known from the suite of music Sergei Prokofiev extracted from his incidental music to the film, the Troika movement even turning up in pop arrangements. The general outline of the plot is well known from the sleeve notes on various recordings. A clerk accidentally generates a non-existent Lieutenant Kizhe in a list to be presented to the tsar. The tsar is interested in this person, and rather than tell him he doesn't exist, the courtiers and officers maintain the pretence that he is real. Kizhe is exiled to Siberia, recalled, promoted, married, promoted again, dies, is given a state funeral, revealed as an embezzler and posthumously demoted to the ranks.<br /><br />I had heard conflicting stories about how the clerk invented Kizhe, involving ink blots and sneezes, but I'd heard the film was lost, so there was no way to find out what happens. Then the film turned up at the Barbican in London as part of their Prokofiev festival. For the record, it turned out that all that happens is that the clerk confuses two words whilst writing an order and turns Kuzhe into Kizhe. As the tsar is in a hurry to see the order, there's no time to correct the mistake.<br /><br />Having gone expecting an historical curiosity, I was pleasantly surprised. The film is very funny, and the audience, myself included, laughed continuously. Although most of it is filmed straight, set mostly in the palace, there are a few 'trick' shots where multiple images appear on the screen. For instance, the tsar's army is represented by a small group, repeated across the screen. Four identical guards perform perfect drill in perfect unison. Two identical servants scrub the floor.<br /><br />One slight drawback was it was very difficult to work out who everyone was. There were two women who might have been the tsar's daughters, or a daughter and a servant or something else. And very few people were named. But all in all, an enjoyable film and I'm surprised it's not seen more often."
# inp = input()
# words = gnlp.single_preprocessing(inp)
# print(node_vectors)
# print(dv_model.infer_vector(words))
# vec = dv_model.infer_vector(words)
# node_vectors = np.append(node_vectors, np.reshape(vec, (1, vec.size)), axis=0)
# print(len(node_vectors))
# inp_a = np.zeros((1, len(A)))
# for key, value in gnlp.my_most_similar(node_vectors, len(node_vectors)-1, 0.55):  ### 0.86 на 15
#     inp_a[0][key] = 1.
# print(np.count_nonzero(inp_a[0]))
# #inp_a[0][13], inp_a[0][2009] = 1., 1.
# inp_x = np.zeros((1, X.shape[1]))
# for i in range(len(words_set)):
#     if words_set[i] in words:
#         inp_x[0][i] = 1.
# print(np.count_nonzero(inp_x[0]))
# # for i in range(100):
# #     r = random.randint(0, X.shape[1]-1)
# #     inp_x[0][r] = 1
# # print(inp_x, len(inp_x[0]))
# dataset = gnlp.MyDataset()
# dataset.add(inp_a, inp_x)
# # print(dataset.graphs[0])
# print(dataset.get_info())
# mask = np.zeros(len(A), dtype=int)
# mask = np.append(mask, 1)
# print(mask, len(mask))
# loader_new = gnlp.SingleLoader(dataset)#, sample_weights=mask)
# pred = model.predict(loader_new.load(), steps=loader_new.steps_per_epoch)
# #pred = model(loader_new, training=False)
# pos, neg = 0, 0
# for p in pred:
#     if p.argmax() == 0:
#         pos += 1
#     elif p.argmax() == 1:
#         neg += 1
# print('POS/NEG: ', pos, neg)
# prediction_result = pred[5002]
# print(pred)
# print(prediction_result)
# prediction_result = prediction_result.argmax()
# print(prediction_result)
############################################


def read_csv(f, c=None, t=np.float64):
    return np.loadtxt(f, usecols=c, delimiter=' ', ndmin=2, dtype=t)


@eel.expose
def gcn_predict(inp):
    #time.sleep(5)
    #return inp
    #######################################
    model = load_model('gcn_model.tf')
    A = np.load('adj_matrix.npy')
    X = np.load('feature_matrix.npy')
    dv_model = Doc2Vec.load('my_doc2vec.model')
    node_vectors = np.load('vectors.npy')
    words_set = np.load('words_set.npy')
    dataset = gnlp.MyDataset()
    #######################################
    words = gnlp.single_preprocessing(inp)
    vec = dv_model.infer_vector(words)
    node_vectors = np.append(node_vectors, np.reshape(vec, (1, vec.size)), axis=0)
    inp_a = np.zeros((1, len(A)))
    for key, value in gnlp.my_most_similar(node_vectors, len(node_vectors) - 1, 0.55):
        inp_a[0][key] = 1.
    inp_x = np.zeros((1, X.shape[1]))
    for i in range(len(words_set)):
        if words_set[i] in words:
            inp_x[0][i] = 1.

    dataset.add(inp_a, inp_x)
    loader_new = gnlp.SingleLoader(dataset)  # , sample_weights=mask)
    pred = model.predict(loader_new.load(), steps=loader_new.steps_per_epoch)
    prediction_prob = pred[-1]
    prediction_result = prediction_prob.argmax()

    print(prediction_result)
    output = f'With probability {max(prediction_prob)*100:.2f}% text is '
    if prediction_result == 1:
        output += 'positive '
    elif prediction_result == 0:
        output += 'negative '
    output += f'(class {prediction_result}).'
    return output

@eel.expose
def gbtree_predict(inp, readcsv=read_csv, method='defaultDense'):
    nFeatures = 15
    nClasses = 2
    maxIterations = 200
    minObservationsInLeafNode = 8
    # input data file
    infile = "train_data.txt"
    # testfile = "test_data.txt"

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
    data = readcsv(infile, range(50), t=np.float32)
    labels = readcsv(infile, range(50, 51), t=np.float32)
    #weights = np.ones((data.shape[0], 1))
    train_result = train_algo.compute(data, labels)
    print('training gbtree model completed...')
    # Now let's do some prediction
    # previous version has different interface
    predict_algo = d4p.gbt_classification_prediction(
        nClasses=nClasses,
    )
    # read test data (with same #features)
    #pdata = readcsv(testfile, range(15), t=np.float32)
    dv_model = Doc2Vec.load('my_doc2vec.model')
    words = gnlp.single_preprocessing(inp)
    pdata = np.reshape(dv_model.infer_vector(words), (-1, 50))
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)
    print('The result is computed')
    # Prediction result provides prediction
    #plabels = readcsv(testfile, range(15, 16), t=np.float32)
    #assert np.count_nonzero(predict_result.prediction - plabels) / pdata.shape[0] < 0.022
    res = int(predict_result.prediction[0][0])
    output = 'Text is '
    if res == 1:
        output += 'positive '
    elif res == 0:
        output += 'negative '
    output += f'(class {res}).'
    return output

eel.start('index.html', mode='chrome', size=(800, 600))
print('Ready to input')
