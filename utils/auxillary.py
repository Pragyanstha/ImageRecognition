import numpy as np
import pickle

def load_data(path):
    data = pickle.load(open(path, 'rb'))
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    return X_train, y_train,X_test, y_test


def prepare_data(path, sets = None):
    X_train, y_train, X_test, y_test = load_data(path)
    if sets:
        raise ValueError('Not implemented')
    
    training_samples = []
    test_samples = []
    mean = np.concatenate(X_train, axis = -1).mean()
    std = np.concatenate(X_train, axis = -1).std()
    for i, label in enumerate(y_train):

        training_samples.append((label, (X_train[i] - mean)/std))
    
    for i, label in enumerate(y_test):
        # mean = np.mean(X_test[i])
        # std = np.std(X_test[i])
        test_samples.append((label, (X_test[i] - mean)/std))

    return training_samples, test_samples


def prepare_data_all(path, sets = None):
    X_train, y_train, X_test, y_test = load_data(path)
    if sets:
        raise ValueError('Not implemented')
    
    training_samples = []
    test_samples = []
    mean = np.concatenate(X_train, axis = -1).mean()
    std = np.concatenate(X_train, axis = -1).std()
    for i in range(len(X_train)//3):
        training_samples.append((y_train[i], (np.concatenate([X_train[i*3 ], X_train[i*3+1], X_train[i*3+2]], axis = -1) - mean)/std))
        #training_samples.append((y_train[i], (np.concatenate([X_train[i*3 ], X_train[i*3+1], X_train[i*3+2]], axis = -1))))
    
    for i in range(len(X_test)//3):
        # mean = np.mean(X_test[i])
        # std = np.std(X_test[i])
        test_samples.append((y_test[i], (np.concatenate([X_test[i*3 ], X_test[i*3+1], X_test[i*3+2]], axis = -1) - mean)/std))
        #test_samples.append((y_test[i], (np.concatenate([X_test[i*3 ], X_test[i*3+1], X_test[i*3+2]], axis = -1))))

    return training_samples, test_samples