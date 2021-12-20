import cupy as cp
from cupy import linalg as LA

def getSubspace(X, N=9):
    n = X.shape[1]
    C = (1/float(n))*(X @ X.T)
    w, v = LA.eigh(C)
    idx = cp.argsort(w)
    vs = v[:,idx[-1:-1-N:-1]]
    return vs

def calcCosineScores(test_sample, train_samples, num_cosines = 1):
    scores = cp.zeros(len(train_samples))
    labels = []
    P = test_sample @ test_sample.T
    for i, (label, train_sample) in enumerate(train_samples):
        Q = train_sample @ train_sample.T
        w, v = LA.eigh(P@Q)
        idx = cp.argsort(w)
        ws = w[idx[-1:-1-num_cosines:-1]]
        scores[i] = cp.sum(ws)
        labels.append(label)

    return scores, labels

