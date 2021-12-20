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
    for i, label in enumerate(y_train):
        training_samples.append((label, X_train[i]))
    
    for i, label in enumerate(y_test):
        test_samples.append((label, X_test[i]))

    return training_samples, test_samples