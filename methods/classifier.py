import numpy as np
from abc import ABC, abstractmethod

class Classifier(ABC):

    def __init__(self, cfg):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
