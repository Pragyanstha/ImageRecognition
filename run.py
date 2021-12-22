import numpy as np
import configargparse
from utils.auxillary import prepare_data, prepare_data_all
import methods
from args import parse
import logging


def run(opt):
    if opt.method == 'msm':
        classifier = methods.MSM(opt.dim_subspace, opt.num_cosines)
    elif opt.method == 'kmsm':
        classifier = methods.KMSM(opt.dim_subspace, opt.num_cosines, opt.sigma, opt.kernel)
    elif opt.method == 'msm_mod':
        classifier = methods.MSM_MOD(opt.dim_subspace, opt.num_cosines)
    
    training_samples, test_samples = prepare_data_all('ytc_py.pkl')

    classifier.train(training_samples)
    accuracy = classifier.evaluate(test_samples)

    logging.debug(f'Final : {accuracy} %')
    

if __name__ == '__main__':
    opt = parse()
    logging.basicConfig(level = logging.DEBUG, filename=f'logs/{opt.expname}.log', filemode='w',format='%(levelname)s - %(message)s')
    logging.debug('STARTED')
    run(opt)