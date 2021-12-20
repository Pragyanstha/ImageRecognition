import numpy as np
import configargparse
from utils.auxillary import prepare_data
import methods
from args import parse
import logging


def run(opt):
    if opt.method == 'msm':
        classifier = methods.MSM(opt.dim_subspace, opt.num_cosines)
    
    training_samples, test_samples = prepare_data('ytc_py.pkl')

    classifier.train(training_samples)
    accuracy = classifier.evaluate(test_samples[:3])

    print(accuracy)
    

if __name__ == '__main__':
    opt = parse()
    logging.basicConfig(level = logging.DEBUG, filename=f'logs/{opt.expname}.log', filemode='w',format='%(name)s - %(levelname)s - %(message)s')
    logging.debug('hello')
    run(opt)