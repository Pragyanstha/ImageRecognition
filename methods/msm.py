import numpy as np
import cupy as cp
from utils.auxillary import prepare_data
from utils.subspaces import getSubspace, calcCosineScores
from methods.classifier import Classifier
import logging

class MSM(Classifier):
    def __init__(self, dim_subspace = 9, num_cosines = 3):
        self.num_cosines = num_cosines
        self.dim_subspace = dim_subspace
        pass

    def train(self, samples):
        self.subspaces = []
        for (y, X) in samples:
            self.subspaces.append((y, getSubspace(cp.asarray(X), self.dim_subspace)))

    def evaluate(self, samples):
        if not hasattr(self, 'subspaces'):
            raise ValueError('Create training subspaces first by calling train()')
        
        accuracy = 0
        for (y, X) in samples:
            test_sample = getSubspace(cp.asarray(X), self.dim_subspace)
            self.scores, self.labels = calcCosineScores(cp.asarray(test_sample), self.subspaces, self.num_cosines)
            highest = cp.argmax(self.scores)
            if self.labels[cp.asnumpy(highest)] == y:
                accuracy += 1
            logging.debug('test')
        
        return accuracy/float(len(samples)) * 100.0


if __name__ == '__main__':

    training_samples, test_samples = prepare_data('ytc_py.pkl')
    msm = MSM()
    msm.train(training_samples)
    acc = msm.evaluate(test_samples[:5])
    print(acc)
    
