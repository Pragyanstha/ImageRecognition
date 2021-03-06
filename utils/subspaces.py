import cupy as cp
from cupy import linalg as LA

def getK_rbf(X, Y, sigma = 0.5, normalize = False):

    X_norm = cp.sum(X ** 2, axis = 0)
    Y_norm = cp.sum(Y ** 2, axis = 0)
    K = cp.exp(-sigma * (X_norm[:, None] + Y_norm[None, :] - 2 * cp.dot(X.T, Y)))

    if normalize:
        # X and Y are the same so
        n = X.shape[1]
        ones = (1/float(n))*cp.ones((n, n))
        K = K - 2*ones@K + ones@K@ones

    return K

def getK_poly(X, Y, sigma = 0.5, normalize = False):

    K = (sigma * X.T @ Y) ** 12

    if normalize:
        # X and Y are the same so
        n = X.shape[1]
        ones = (1/float(n))*cp.ones((n, n))
        K = K - 2*ones@K + ones@K@ones

    return K

def getKernelBasis(X, N = 9):
    w, v = LA.eigh(X)
    idx = cp.argsort(w)
    vs = v[:,idx[-1:-1-N:-1]]
    return vs


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

def calcCosineScores2(test_sample, train_samples, num_cosines = 1):
    scores = cp.zeros(len(train_samples))
    labels = []
    for i, (label, train_sample) in enumerate(train_samples):
        C = test_sample.T @ train_sample @ train_sample.T @ test_sample
        w, v = LA.eigh(C)
        idx = cp.argsort(w)
        ws = w[idx[-1:-1-num_cosines:-1]]
        scores[i] = cp.sum(ws)
        labels.append(label)

    return scores, labels    

def calcCosineScores3(test_sample, train_samples, num_imgs_test, num_imgs_train, num_cosines = 1):
    scores = cp.zeros(len(train_samples))
    labels = []
    for i, (label, train_sample) in enumerate(train_samples):
        C = test_sample.T @ train_sample @ train_sample.T @ test_sample
        w, v = LA.eigh(C)
        idx = cp.argsort(w)
        ws = w[idx[-1:-1-num_cosines:-1]]
        scores[i] = cp.sum(ws)
        
        labels.append(label)

    return scores, labels    

def calcAngleScores2(test_sample, train_samples, num_cosines = 1):
    scores = cp.zeros(len(train_samples))
    labels = []
    for i, (label, train_sample) in enumerate(train_samples):
        C = test_sample.T @ train_sample @ train_sample.T @ test_sample
        w, v = LA.eigh(C)
        idx = cp.argsort(w)
        ws = w[idx[-1:-1-num_cosines:-1]]
        scores[i] = cp.sum(cp.arccos(ws) ** 2)
        labels.append(label)

    return scores, labels    

def calcKernelCosineScores2(train_bases, test_basis,X, Y, num_cosines = 1):
    scores = cp.zeros(len(train_bases))
    labels = []
    for i, (label, train_basis) in enumerate(train_bases):
        K = getK_rbf(X[i], Y)
        C = train_basis.T @ K @ test_basis
        C = C @ C.T
        w, v = LA.eigh(C)
        idx = cp.argsort(w)
        ws = w[idx[-1:-1-num_cosines:-1]]
        scores[i] = cp.sum(ws)
        #scores[i] = cp.sum(cp.arccos(ws) ** 2)
        labels.append(label)

    return scores, labels    

def calcMODScore(test_sample, train_samples, dim_diffspace = 9,  num_cosines = 1):   
    # C = test_sample.T @ train_sample @ train_sample.T @ test_sample
    # w, v = LA.eigh(C)
    # idx = cp.argsort(w)
    # ws = w[idx[-1:-1-num_cosines:-1]]
    # score = 1/cp.sum(ws)

    # Generate the G matrix
    G = cp.zeros((train_samples[0].shape[0], train_samples[0].shape[0]))

    for S in train_samples:
        G += S@S.T
    
    w, v = LA.eigh(G)
    idx = cp.argsort(w)
    vs = v[:, idx[0:dim_diffspace]]

    # The Difference Subpace Projection Matrix
    D = vs @ vs.T

    # Calculate projected subspace bases
    test_sample_projected = D @ test_sample
    test_sample_projected = test_sample_projected * cp.linalg.norm(test_sample_projected, axis = 0, keepdims=True)
    test_sample_projected, dump = cp.linalg.qr(test_sample_projected)
    scores = cp.zeros(len(train_samples))
    for i, train_sample in enumerate(train_samples):
        train_sample_projected, R = cp.linalg.qr(D @ S)
        C = test_sample.T @ train_sample @ train_sample.T @ test_sample
        w, v = LA.eigh(C)
        idx = cp.argsort(w)
        ws = w[idx[-1:-1-num_cosines:-1]]
        scores[i] = cp.sum(ws)


    return scores