import numpy as np
import cupy as cp
from utils.auxillary import prepare_data
from utils.subspaces import getK_rbf, getK_poly, getKernelBasis, calcKernelCosineScores2, calcCosineScores2, calcMODScore
from methods.classifier import Classifier
import logging

class MSM_MOD(Classifier):
    def __init__(self, dim_subspace = 9, num_cosines = 3, sigma = 0.5, kernel='rbf'):
        self.num_cosines = num_cosines
        self.dim_subspace = dim_subspace
        self.sigma = sigma
        if kernel == 'rbf':
            self.getK = getK_rbf
        elif kernel == 'poly':
            self.getK = getK_poly


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
            
            
            # Creating top 5 candidates score
            top5 = cp.asnumpy(cp.argsort(self.scores)[-1:-6:-1])
            candidates = np.array(self.labels, dtype=object)[top5]
            if y in candidates:
                top5_accuracy += 1

            cand_X = [self.train_samples[idx] for idx in top5]
            cand_y = [self.labels[idx] for idx in top5]
            mod_score = cp.zeros(5)
            # mod_score[0] = calcMODScore(cp.concatenate([cand_X[0], cp.asarray(X)], axis=-1), cp.concatenate([cand_X[1], cand_X[2], cand_X[3], cand_X[4]], axis=-1))
            # mod_score[1] = calcMODScore(cp.concatenate([cand_X[1], cp.asarray(X)], axis=-1), cp.concatenate([cand_X[0], cand_X[2], cand_X[3], cand_X[4]], axis=-1))
            # mod_score[2] = calcMODScore(cp.concatenate([cand_X[2], cp.asarray(X)], axis=-1), cp.concatenate([cand_X[1], cand_X[0], cand_X[3], cand_X[4]], axis=-1))
            # mod_score[3] = calcMODScore(cp.concatenate([cand_X[3], cp.asarray(X)], axis=-1), cp.concatenate([cand_X[1], cand_X[2], cand_X[0], cand_X[4]], axis=-1))
            # mod_score[4] = calcMODScore(cp.concatenate([cand_X[4], cp.asarray(X)], axis=-1), cp.concatenate([cand_X[1], cand_X[2], cand_X[3], cand_X[0]], axis=-1))

            mod_score[0] = calcMODScore(cp.asarray(X), cand_X[0])
            mod_score[1] = calcMODScore(cp.asarray(X), cand_X[1])
            mod_score[2] = calcMODScore(cp.asarray(X), cand_X[2])
            mod_score[3] = calcMODScore(cp.asarray(X), cand_X[3])
            mod_score[4] = calcMODScore(cp.asarray(X), cand_X[4])

            highest = cp.argmax(mod_score + self.scores[top5])
            
            if  y == cand_y[cp.asnumpy(highest)]:
                accuracy += 1
            # if self.labels[cp.asnumpy(highest)] == y:
            #     accuracy += 1
            logging.debug(f'Test Case - {i}/{len(samples)} \nTop 1 : {accuracy/float(i+1) * 100} % \nTop 5 : {top5_accuracy/float(i+1) * 100}')
        
        return accuracy/float(len(samples)) * 100.0