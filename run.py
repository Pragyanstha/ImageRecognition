import sys
import numpy as np
import configargparse
from utils.auxillary import prepare_data, prepare_data_all
import methods
from args import parse
import logging
import optuna


def run(opt):

    training_samples, test_validation_samples = prepare_data_all('ytc_py.pkl', opt.mode)

    if opt.mode == 'test':
        if opt.method == 'msm':
            classifier = methods.MSM(opt.dim_subspace, opt.num_cosines)
        elif opt.method == 'kmsm':
            classifier = methods.KMSM(opt.dim_subspace, opt.num_cosines, opt.sigma, opt.kernel)
        elif opt.method == 'kmsm_mod':
            classifier = methods.KMSM_MOD(opt.dim_subspace, opt.num_cosines, opt.sigma, opt.kernel, opt.dim_diffspace)
        
        classifier.train(training_samples)  
        top1_accuracy, top5_accuracy = classifier.evaluate(test_validation_samples)
        logging.debug(f'----- Result ------\nTop 1 {top1_accuracy} %\nTop 5 {top5_accuracy} %')
        return 0


    elif opt.mode == 'validation':
        def objective(trial):
            dim_subspace = trial.suggest_int('dim_subspace', 3, 30)
            num_cosines =  trial.suggest_int('num_cosines', 3, dim_subspace)
            if opt.method == 'msm':
                classifier = methods.MSM(dim_subspace, num_cosines)
            elif opt.method == 'kmsm':
                sigma = trial.suggest_float('sigma', 1e-03, 1.0, log=True)
                kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
                classifier = methods.KMSM(dim_subspace, num_cosines, sigma, kernel)
            elif opt.method == 'kmsm_mod':
                sigma = trial.suggest_float('sigma', 1e-03, 1.0, log=True)
                kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
                dim_diffspace = trial.suggest_int('dim_diffspace', 3, 30)
                classifier = methods.KMSM_MOD(dim_subspace, num_cosines, sigma, kernel, dim_diffspace)

            classifier.train(training_samples)  
            top1_accuracy, top5_accuracy = classifier.evaluate(test_validation_samples)
            return -top1_accuracy
        
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        storage_name = "sqlite:///db/{}.db".format(opt.expname)
        study = optuna.create_study(study_name=opt.expname, storage=storage_name, load_if_exists=True)
        study.optimize(objective, n_trials=50)  # Invoke optimization of the objective function.
        #logging.debug(f'----- Result ------\nTop 1 {top1_accuracy} %\nTop 5 {top5_accuracy} %')
        print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
        return 0

    
    

if __name__ == '__main__':
    opt = parse()
    logging.basicConfig(level = logging.DEBUG, filename=f'logs/{opt.expname}.log', filemode='w',format='%(message)s')
    logging.debug('STARTED')
    run(opt)