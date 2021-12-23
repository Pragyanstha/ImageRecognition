import numpy as np
import cupy as cp
from utils.auxillary import prepare_data
from utils.subspaces import getSubspace, calcCosineScores, calcCosineScores2
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
        top5_accuracy = 0
        for i, (y, X) in enumerate(samples):
            test_sample = getSubspace(cp.asarray(X), self.dim_subspace)
            self.scores, self.labels = calcCosineScores2(cp.asarray(test_sample), self.subspaces, self.num_cosines)
            highest = cp.argmax(self.scores)
            top5 = cp.argsort(self.scores)[-1:-6:-1]
            candidates = np.array(self.labels, dtype=object)[cp.asnumpy(top5)]
            if y in candidates:
                top5_accuracy += 1

            if self.labels[cp.asnumpy(highest)] == y:
                accuracy += 1
            logging.debug(f'Test Case - {i}/{len(samples)} \nTop 1 : {accuracy/float(i+1) * 100} % \nTop 5 : {top5_accuracy/float(i+1) * 100}')
        
        return accuracy/float(len(samples)) * 100.0, top5_accuracy/float(len(samples)) * 100


if __name__ == '__main__':

    training_samples, test_samples = prepare_data('ytc_py.pkl')
    msm = MSM()
    msm.train(training_samples)
    acc = msm.evaluate(test_samples[:5])
    print(acc)
    
