import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.linalg import lstsq
import numpy as np
import math
import sys
import statistics
import smooth_sens_median as ssm

# Sample from Laplace log-normal distribution
def laplaceLogNormalRV(sigma):
    """
    :param sigma: LLN parameter
    :returns: random variable sampled from LLN(sigma)
    """
    X = np.random.laplace()
    Y = np.random.normal()
    Z = X * math.exp(sigma * Y)
    return Z

# Sample from uniform log-normal distribution
def uniformLogNormalRV(sigma):
    """
    :param sigma: ULN parameter
    :returns: random variable sampled from ULN(sigma)
    """
    X = np.random.uniform(-1, 1)
    Y = np.random.normal()
    Z = X * math.exp(sigma * Y)
    return Z

# Sample from Student's T distribution with d degrees of freedom
def studentsTRV(d):
    """
    :param d: degrees of freedom (should be >= 1)
    :returns: random variable sampled from T(d)
    """
    X = np.random.normal(size=d+1)
    Z = X[0] / math.sqrt((sum(X[1:]**2))/d)
    return Z

# Sample from arsinh-normal distribution with parameter sigma
def arsinhNormalRV(sigma):
    """
    :param sigma: arsinh normal parameter
    :returns: random variable sampled from arsinh-normal(sigma)
    """
    Y = np.random.normal()
    Z = (1.0/sigma)*math.sinh(sigma*Y)
    return Z

# Compute .5eps^2-CDP smooth sens median for bounded data using laplace log-normal noise distribution.
# Wrapper for DP median using laplace log-normal noise addition.
def dpMedianLLN(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
    """
    :param x: List or numpy array of real numbers
    :param lower_bound: Lower bound on values in x. 
    :param upper_bound: Upper bound on values in x.
    :param epsilon: Desired value of epsilon.
    :param hyperparameters: dictionary that contains:
        :param beta: Smoothing parameter (positive float). Defaults to 0.1.
        :param smooth_sens: beta-smooth sensitivity of the median. Must be included.
    :param num_trials: Number of fresh DP median estimates to generate using dataset x.
    :return: List of 1/2 epsilon^2-CDP estimates for the median of x.
    """
    beta = hyperparameters['beta'] if ('beta' in hyperparameters) else default_beta
    assert beta >= 0.0
    
    smooth_sens = ssm.smooth_sens_median(x, beta, lower_bound, upper_bound)[0]

    results = []
    for i in range(num_trials):
        true_median = np.median(x)
        sigma = max(2*beta/epsilon, 1/2)
        Z = laplaceLogNormalRV(sigma)
        s = math.exp(-(3/2)*(sigma**2)) * (epsilon - (abs(beta)/abs(sigma)))
        res = true_median + (1/s)*smooth_sens*Z
        results.append(res)

    return results

# Compute .5eps^2-CDP smooth sens median for bounded data using uniform log-normal noise distribution.
# Wrapper for DP median using uniform log-normal noise addition.
def dpMedianULN(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
    """
    :param x: List or numpy array of real numbers
    :param lower_bound: Lower bound on values in x. 
    :param upper_bound: Upper bound on values in x.
    :param epsilon: Desired value of epsilon.
    :param hyperparameters: dictionary that contains:
        :param beta: Smoothing parameter (positive float). Defaults to 0.1.
        :param smooth_sens: beta-smooth sensitivity of the median. Must be included.
    :param num_trials: Number of fresh DP median estimates to generate using dataset x.
    :return: List of 1/2 epsilon^2-CDP estimates for the median of x.
    """
    default_beta = 0.9
    beta = hyperparameters['beta'] if ('beta' in hyperparameters) else default_beta
    assert beta >= 0.0
    
    smooth_sens = ssm.smooth_sens_median(x, beta, lower_bound, upper_bound)[0]

    results = []
    for i in range(num_trials):
        true_median = np.median(x)
        sigma = math.sqrt(2)
        Z = laplaceLogNormalRV(sigma)
        s = math.exp(-(3/2)*(sigma**2)) * math.sqrt(math.pi * sigma**2 / 2) * (epsilon - (abs(beta)/abs(sigma)))
        res = true_median + (1/s)*smooth_sens*Z
        results.append(res)

    return results

# Compute .5eps^2-CDP smooth sens median for bounded data using student's T noise distribution.
# Wrapper for DP median using student's T noise addition.
def dpMedianStudentsT(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
    """
    :param x: List or numpy array of real numbers
    :param lower_bound: Lower bound on values in x. 
    :param upper_bound: Upper bound on values in x.
    :param epsilon: Desired value of epsilon.
    :param hyperparameters: dictionary that contains:
        :param beta: Smoothing parameter (positive float). Defaults to 0.1.
        :param smooth_sens: beta-smooth sensitivity of the median. Must be included.
        :param d: degrees of freedom for Student's T distribution. Defaults to 3.
    :param num_trials: Number of fresh DP median estimates to generate using dataset x.
    :return: List of 1/2 epsilon^2-CDP estimates for the median of x.
    """
    default_d = 3
    default_beta_prop = 0.5
    d = hyperparameters['d'] if ('d' in hyperparameters) else default_d
    beta_prop = hyperparameters['beta_prop'] if ('beta_prop' in hyperparameters) else default_beta_prop
    assert d > 0
    assert beta_prop < 1.0 and beta_prop > 0.0

    beta = float(epsilon)* beta_prop /float(d+1)
    
    smooth_sens = ssm.smooth_sens_median(x, beta, lower_bound, upper_bound)[0]

    results = []
    for i in range(num_trials):
        true_median = np.median(x)
        Z = studentsTRV(d)
        s = 2 * math.sqrt(d) * (epsilon - abs(beta) * (d+1)) / (d+1)
        res = true_median + (1/s)*smooth_sens*Z
        results.append(res)
    return results

# Compute .5eps^2-CDP smooth sens median for bounded data using arsinh-normal noise distribution.
# Wrapper for DP median using arsinh normal noise addition.
def dpMedianArsinhNormal(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
    """
    :param x: List or numpy array of real numbers
    :param lower_bound: Lower bound on values in x. 
    :param upper_bound: Upper bound on values in x.
    :param epsilon: Desired value of epsilon.
    :param hyperparameters: dictionary that contains:
        :param beta: Smoothing parameter (positive float). Defaults to 0.1.
        :param smooth_sens: beta-smooth sensitivity of the median. Must be included.
    :param num_trials: Number of fresh DP median estimates to generate using dataset x.
    :return: List of 1/2 epsilon^2-CDP estimates for the median of x.
    """
    beta = hyperparameters['beta'] if ('beta' in hyperparameters) else default_beta
    assert beta >= 0.0
    
    smooth_sens = ssm.smooth_sens_median(x, beta, lower_bound, upper_bound)[0]

    results = []
    for i in range(num_trials):
        true_median = np.median(x)
        sigma = 2/math.sqrt(3)
        Z = arsinhNormalRV(sigma)
        s = (6 * sigma / (4 + 3 * sigma**2)) * (epsilon -  math.sqrt(abs(beta) * ((abs(beta)/(sigma**2)) + (1/sigma) + 2)))
        res = true_median + (1/s)*smooth_sens*Z
        results.append(res)
    return results

# Testing:
# x = [0, 1, 2, 3, 4, 5, 6]
# lower_bound = 0
# upper_bound = 5
# epsilon = 10
# beta = 0.9
# d = 3
# smooth_sens = smooth_sens_median_slow(x, beta, lower_bound, upper_bound)[0]
# hyperparameters = {'smooth_sens': smooth_sens, 'beta': beta, 'd':3}
# num_trials = 5
# print(dpMedianULN(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials))
