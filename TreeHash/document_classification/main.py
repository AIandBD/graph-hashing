import time
from tqdm import tqdm
import numpy as np
import itertools
from collections import Counter
from itertools import chain
from sklearn import svm
from sklearn.svm import SVC
import scipy.spatial.distance as spd
from sklearn.model_selection import KFold
from preprocess import preprocess
C_PRIME =  10000000000037

def minhash(tree, feature, K,  M, D):
    N = len(tree)
    fingerprints = np.zeros((N, K))
    hash_parameters = np.random.randint(1, C_PRIME, (K, 2))
    for n in tqdm(range(0, N)):
        nodes = tree[n][D]
        counter = Counter(feature[nodes].indices)
        feature_id = np.array([i for i in counter if counter[i] > M]).reshape(-1, 1)
        feature_id_num = feature_id.shape[0]
        k_hash = np.mod(
            np.dot(np.transpose(np.array([feature_id])), np.array([np.transpose(hash_parameters[:, 1])])) +
            np.dot(np.ones((feature_id_num, 1)), np.array([np.transpose(hash_parameters[:, 0])])),
            C_PRIME)
        min_position = np.argmin(k_hash, axis=1)
        fingerprints[n, :] = feature_id[min_position].reshape((1, -1))
    return fingerprints


def classification(vectors, labels, kernel):
    accs = []
    tt = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(vectors):
        t1 = time.time()
        trainKernel, trainLabels = kernel[train_index][:, train_index], labels[train_index]
        testKernel, testLabels = kernel[test_index][:, train_index], labels[test_index]
        svmModel = svm.SVC(kernel='precomputed').fit(trainKernel, trainLabels)
        preds = svmModel.predict(testKernel)
        acc = np.sum(preds == testLabels) / len(testLabels)
        accs.append(acc)
        et = time.time()-t1
        tt.append(et)
    accs = np.array(accs)
    mean_acc = round(np.mean(accs) * 100, 2)
    std_acc = round(np.std(accs) * 100, 2)
    mean_tt = round(np.mean(tt), 2)
    std_tt = round(np.std(tt), 2)
    return mean_acc, std_acc, mean_tt, std_tt


if __name__ == '__main__':
    #datasets: cora,citeseer,dblp,NELL
    data = 'NELL'
    tree, feature, labels = preprocess(data)
    st1 = time.time()
    # K=200,M=4,D=1 for cora,citeseer,dblp;
    # K=200,M=1,D=1 for NELL
    vectors = minhash(tree, feature, K=200, M=1, D=1)
    embedding_time = time.time()-st1
    gram_matrix = 1 - spd.squareform(spd.pdist(vectors, 'hamming'))
    time1 = time.time() - st1
    mean_acc, std_acc, cla_time, std_time = classification(vectors, labels, gram_matrix)
    np.savez('./results/' + data + "_embeddings.npz", embeddings=vectors, embedding_time=round(embedding_time, 2),
             labels=labels)
    with open('./results/' + 'result.txt', 'a+') as f:
        # f.writelines("dataset"+'\t'+"mean_acc"+'\t'+"std_acc"+'\t'+"mean_time"+'\t'+"std_time"+'\n')
        f.writelines(data + '\t' + str(mean_acc) + '\t' + str(std_acc) + '\t' + str(cla_time) + '\t' + str(std_time) + '\n')
    print("total_accuracy:", mean_acc)
    print("total_time:", time1 + cla_time)



