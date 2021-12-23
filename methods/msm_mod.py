import numpy as np
import cupy as cp
from utils.auxillary import prepare_data
from utils.subspaces import getSubspace, calcCosineScores, calcCosineScores2, calcCosineScores3, calcMODScore
from methods.classifier import Classifier
import logging

class MSM_MOD(Classifier):
    def __init__(self, dim_subspace = 9, dim_diffspace = 9, num_cosines = 3):
        self.num_cosines = num_cosines
        self.dim_subspace = dim_subspace
        self.dim_diffspace = dim_diffspace
        pass

    def train(self, samples):
        self.subspaces = []
        self.num_imgs = []
        self.train_samples = []
        for (y, X) in samples:
            self.subspaces.append((y, getSubspace(cp.asarray(X), self.dim_subspace)))
            self.train_samples.append(cp.asarray(X))
            self.num_imgs.append(X.shape[1])

    def evaluate(self, samples):
        if not hasattr(self, 'subspaces'):
            raise ValueError('Create training subspaces first by calling train()')
        
        accuracy = 0
        top5_accuracy = 0
        for i, (y, X) in enumerate(samples):
            test_sample = getSubspace(cp.asarray(X), self.dim_subspace)
            self.scores, self.labels = calcCosineScores3(cp.asarray(test_sample), self.subspaces, X.shape[1], self.num_imgs, self.num_cosines)
            
            # Creating top 5 candidates score
            top5 = cp.asnumpy(cp.argsort(self.scores)[-1:-6:-1])
            candidates = np.array(self.labels, dtype=object)[top5]

            cand_Subspace = [getSubspace(self.train_samples[idx], self.dim_subspace) for idx in top5]
            cand_Labels = [self.labels[idx] for idx in top5]

            mod_scores = calcMODScore(getSubspace(cp.asarray(X), self.dim_subspace), cand_Subspace, self.dim_diffspace, num_cosines=self.num_cosines)
            highest = cp.argmax(mod_scores)
    
            if y in candidates:
                top5_accuracy += 1

            if y == cand_Labels[cp.asnumpy(highest)]:
                accuracy += 1

            logging.debug(f'Test Case - {i}/{len(samples)} \nTop 1 : {accuracy/float(i+1) * 100} % \nTop 5 : {top5_accuracy/float(i+1) * 100}')
        
        return accuracy/float(len(samples)) * 100.0, top5_accuracy/float(len(samples)) * 100

   # For illustration purpose
    def evalulate_single(self, X):
        if not hasattr(self, 'subspaces'):
            raise ValueError('Create training subspaces first by calling train()')
        
        test_sample = getSubspace(cp.asarray(X), self.dim_subspace)
        self.scores, self.labels = calcCosineScores3(cp.asarray(test_sample), self.subspaces, X.shape[1], self.num_imgs, self.num_cosines)


        # Creating top 5 candidates score
        top5 = cp.asnumpy(cp.argsort(self.scores)[-1:-6:-1])
        candidates = np.array(self.labels, dtype=object)[top5]

        cand_Subspace = [getSubspace(self.train_samples[idx], self.dim_subspace) for idx in top5]
        cand_Labels = [self.labels[idx] for idx in top5]
        cand_Imgs = [self.train_samples[idx] for idx in top5]

        mod_scores = calcMODScore(getSubspace(cp.asarray(X), self.dim_subspace), cand_Subspace, dim_diffspace=9, num_cosines=self.num_cosines)

        return cand_Labels, self.scores[top5], mod_scores, cand_Imgs

if __name__ == '__main__':

    training_samples, test_samples = prepare_data('ytc_py.pkl')
    msm = MSM()
    msm.train(training_samples)
    acc = msm.evaluate(test_samples[:5])
    print(acc)
    
