import numpy as np
import cupy as cp
from utils.auxillary import prepare_data
from utils.subspaces import getK_rbf, getK_poly, getKernelBasis, calcKernelCosineScores2
from methods.classifier import Classifier
import logging

class KMSM(Classifier):
    def __init__(self, dim_subspace = 9, num_cosines = 3, sigma = 0.5, kernel='rbf'):
        self.num_cosines = num_cosines
        self.dim_subspace = dim_subspace
        self.sigma = sigma
        if kernel == 'rbf':
            self.getK = getK_rbf
        elif kernel == 'poly':
            self.getK = getK_poly

        print(self.getK)

    def train(self, samples):
        self.train_bases = []
        self.train_samples = []
        for (y, X) in samples:
            K = self.getK(cp.asarray(X), cp.asarray(X), self.sigma, normalize=True)
            self.train_bases.append((y, getKernelBasis(K, self.dim_subspace)))
            self.train_samples.append(cp.asarray(X))

    def evaluate(self, samples):
        if not hasattr(self, 'train_bases'):
            raise ValueError('Create training subspaces first by calling train()')
        
        accuracy = 0
        top5_accuracy = 0
        for i, (y, X) in enumerate(samples):
            K = self.getK(cp.asarray(X), cp.asarray(X), self.sigma)
            test_basis = getKernelBasis(K, self.dim_subspace)
            test_sample = cp.asarray(X)
            self.scores, self.labels = calcKernelCosineScores2(self.train_bases, test_basis, self.train_samples,test_sample, self.num_cosines)
            highest = cp.argmax(self.scores)
            
            # Creating top 5 candidates score
            top5 = cp.argsort(self.scores)[-1:-6:-1]
            candidates = np.array(self.labels, dtype=object)[cp.asnumpy(top5)]
            if y in candidates:
                top5_accuracy += 1


            if self.labels[cp.asnumpy(highest)] == y:
                accuracy += 1
            logging.debug(f'Test Case - {i}/{len(samples)} \nTop 1 : {accuracy/float(i+1) * 100} % \nTop 5 : {top5_accuracy/float(i+1) * 100}')
        
        return accuracy/float(len(samples)) * 100.0