from scipy.special import erfinv
from scipy.special import binom
import scipy.stats as st
import numpy as np
from math import sqrt

def getConfIntervalIndices(n, alpha):
	"""
	Determines the indices of the values that correspond to the min and max of the 1-alpha confidence interval for the median on 
	a data set of size n. (For large n.)
	
	See notes/publicCI.pdf for a derivation of this calculation. 
	
	:param n : number of data points in the sample
	:param alpha: parameter for a 1-alpha confidence interval
	:return: [i,j] where i and j are indices of min and max of the 1-alpha confidence interval for the median on a data set of size n
	"""
	#taking floor and ceiling of results and reindexing so first index is 0
	i = int(n/2 + sqrt(n/2) * erfinv(alpha - 1)) # normal approximation
	# i = int(binom.ppf(alpha/2., n, 0.5))
	j = n-i
	return [i-1,j] 

def getConfInterval(x, n, alpha):
	"""
	Returns the min and max of the 1-alpha confidence interval for the median on x.
	
	:param n : number of data points in the sample
	:param alpha: parameter for a 1-alpha confidence interval
	:return: [i,j] where i and j are the min and max of the 1-alpha confidence interval for the median on x.
	"""
	y = sorted(x)
	[i,j] = getConfIntervalIndices(n, alpha)
	return(y[i], y[j])

def getConfIntervalQuantiles(n, alpha):
	"""
	Returns the min and max quantiles of the 1-alpha confidence interval.
	
	:param n : number of data points in the sample
	:param alpha: parameter for a 1-alpha confidence interval
	:return: [i,j] where i and j are the min and max of the 1-alpha confidence interval for the median on x.
	"""
	[i,j] = getConfIntervalIndices(n, alpha)
	return(i/float(n), j/float(n))

def computeTheoreticalCoverage(n, lower_quantile, upper_quantile):
	prob_sum = 0.0
	for i in range(int(n*lower_quantile), int(n*upper_quantile)+1):
		prob = binom(n, i) * (0.5)**(i) * (0.5)**(n-i) if n < 800 else st.norm.pdf(i, loc=n*0.5, scale=np.sqrt(n*0.25))
		prob_sum += prob
	return prob_sum


def getLognormConfInterval(x, n, s, loc, scale, alpha):
	"""
	Returns the min and max of the 1-alpha confidence interval for the median on x under assumption that data is lognormal.
	
	:param n : number of data points in the sample
	:param alpha: parameter for a 1-alpha confidence interval
	:return: [i,j] where i and j are the min and max of the 1-alpha confidence interval for the median on x.
	"""
	# print("min x:", min(x), "max x:", max(x))
	# print("s:", s, "scale:", scale)
	y = np.log((x - loc)/s)
	y_std = np.std(y)
	y_center = np.mean(y) 
	multiplier =  y_std / sqrt(n)  #sqrt( (y_std**2 / n) + (y_std**4/(2*(n-1))) ) #
	z_val = st.norm.ppf(1.-alpha/2.0)
	l_norm = y_center - multiplier* z_val
	u_norm = y_center + multiplier* z_val
	l_lognorm = (np.exp(l_norm)*s + loc)
	r_lognorm = (np.exp(u_norm)*s + loc)
	# print("l, u:", l_norm, u_norm)
	# print("exp l, u:", l_lognorm, r_lognorm)
	return(l_lognorm, r_lognorm)

def nonprivCIsLognormal(x, lower_bound, upper_bound, eps, hyperparameters, num_trials):
	"""
	Wrapper for non-priv lognormal confidence interval that matches private wrappers, eg. functionname(x_clipped, lower_bound, upper_bound, eps, hyperparameters, num_trials)
	:param n : number of data points in the sample
	:param alpha: parameter for a 1-alpha confidence interval
	:return: [i,j] where i and j are the min and max of the 1-alpha confidence interval for the median on x.
	"""
	alpha = hyperparameters['alpha'] if ('alpha' in hyperparameters) else None
	s = hyperparameters['s'] if ('s' in hyperparameters) else 1.0
	loc = hyperparameters['loc'] if ('loc' in hyperparameters) else 0.0
	scale = hyperparameters['scale'] if ('scale' in hyperparameters) else 1.0
	n = len(x)
	results = [getLognormConfInterval(x, n, s, loc, scale, alpha) for i in range(num_trials)]
	return results


