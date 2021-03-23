import os
import matplotlib.pyplot as plt
import numpy as np
import statistics as stats
import pandas
import common

from publicCI import *

data_path = r'data'
res_path = r'results'


def main(name, dataset_name, functionname, hyperparameters, num_trials, rho, lower_bound, upper_bound, alpha=0.05, beta=0.01, 
    dir_path=data_path, res_path=res_path, num_datasets=100):
    """
    This is a wrapper function that runs the given method on the child's income data on every tract in IL, 
    and saves the results as a num_tracts length array of num_trials length arrays.

    It takes hyperparameters as a vector of inputs. Each function will need to be defined to take its hyperparameters as a vector input.

    :param name: what you want the function to named in the saved .npy file.
    :param dataset_name: name of the input dataset.
    :param functionname: the name of the actual function you are calling, together with where it is found.
    :param hyperparameters: vector of hyperparameters needed for that algorithm
    :param num_trials: number of trials to run
    :param rho: CDP privacy parameter
    :param lower_bound: lower bound of range
    :param upper_bound: upper bound of range
    :param alpha: total failure probability (default 0.05)
    :param beta: private CI failure probability (default 0.01); nonprivate failure probability is alpha-beta

    :return: It doesn't output anything but saves a file that contains a header containing experiment parameters, and a num_tracts 
    length array of arrays that contains (a, len(x), nonprivate empirical median, public_ci) where 
    a is a num_trials length array containing the results of num_trials runs of the algorithm, and public_ci is the (1-alpha-beta) percent
    nonprivate confidence interval on the empirical median.

    Note that the saved object is a numpy array object that contains the header at index 0 and the data at index 1. 
    To load data, will need to set allow_pickle = True in the np.load command, which is required for object arrays.
    An alternative approach would be to save the data instead as a text file with a header, which would omit the need
    to pickle. The advantage of this would be that pickling can introduce security concerns and may operate differently
    on different computers. But npy arrays are more efficiently stored, so unless the pickling causes issues we can
    keep it for now. 
    """

    eps = common.rho_to_eps(rho)
    i = 0

    assert alpha > beta and alpha < 1.0
    assert beta > 0.0 and beta < 1.0
    assert lower_bound <= upper_bound

    allresults = []
    for files in os.walk(dir_path): 
        for file in files[2]:  
            if file.startswith(dataset_name) and i < num_datasets: 
                print("dataset", i, "of", num_datasets, dataset_name, str(file))
                i += 1
                x = np.load(dir_path+'/'+str(file))
                n = len(x)
                public_ci = getConfInterval(x, n, alpha) # Compare to non-private alpha quantiles
                if i < 20:
                    hyperparameters['save_path'] = res_path+'/'+name+'_cdf_'+str(file) # for saving noisy cdfs
                nonpriv_beta = (alpha-beta)/(1.-beta) # compute nonprivate failure probability
                lower_quantile, upper_quantile = getConfIntervalQuantiles(n, nonpriv_beta) # private algs take in non-private quantiles
                hyperparameters['lower_quantile'] = lower_quantile
                hyperparameters['upper_quantile'] = upper_quantile
                x_clipped = np.clip(x,lower_bound,upper_bound)

                allresults.append([functionname(x_clipped, lower_bound, upper_bound, eps, hyperparameters, num_trials), n, stats.median(x), public_ci])

    header = [('lower_bound', lower_bound), \
		('upper_bound', upper_bound), \
		('epsilon', eps), \
		('hyperparameters', hyperparameters), \
		('num_trials', num_trials)]

    # print(header)
    # print(allresults)
    # print([header, allresults])

    savePath = '%s/%s_%s.npy' % (res_path, dataset_name, name)
    print(savePath)
    np.save(savePath, np.array([header,allresults]))


